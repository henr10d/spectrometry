import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.chemistry import neutral_mass,formula_mass,identity,deduplicate,adduct_parameters
from src.metric import evaluate
from src.inference import check_submission,Retriever
from src.validation import assignments,assert_disjoint
from src.spectra import modified_cosine,clean_peaks
from src.candidates import CandidateDB
from scripts.enrich_coconut_prior import enrich

class Tests(unittest.TestCase):
    def test_np_prior_keeps_exact_spectral_evidence_and_respects_exclusion(self):
        mass=formula_mass('C2H6O')
        ethanol=identity('CCO')[1];ether=identity('COC')[1]
        frame=pd.DataFrame({'mass':[mass,mass], 'key':[ethanol,ether],
                            'smiles':['CCO','COC'], 'coconut_annotation':[0.,5.],
                            'coconut_records':[0,3]})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'c.parquet';frame.to_parquet(path,index=False)
            r=Retriever.__new__(Retriever);r.config={};r.db=CandidateDB(path)
            r.meta=pd.DataFrame({'key':[ethanol],'ingest_lib':['reference']})
            r.ids=np.array([r.db.key_to_id[ethanol]],dtype=np.int32)
            r.order=np.array([0]);r.sorted_ids=r.ids.copy()
            r.mode=np.array(['positive']);r.adduct=np.array(['[M+H]+'])
            r.peaks=np.stack([clean_peaks([20.,30.],[1.,1.])])
            q=pd.DataFrame({'precursor_mz':[mass+adduct_parameters('[M+H]+')[2]],
                            'adduct':['[M+H]+'],'ionization_mode':['positive'],
                            'ms2_mzs':[[20.,30.]],'ms2_normalized_intensities':[[1.,1.]]})
            ids,_=r.predict(q,method='np_prior')
            self.assertEqual(r.db.key[ids[0]],ethanol)
            ids,_=r.predict(q,method='np_prior',excluded_keys={ethanol})
            self.assertEqual(r.db.key[ids[0]],ether)
            # A provider's noncanonical key must not bypass identity exclusion.
            r.db.key[r.db.key_to_id[ether]]='PROVIDERKEYXXX'
            ids,info=r.predict(q,method='np_prior',excluded_keys={ethanol},
                               candidate_excluded_keys={ether},canonical_oracle=True)
            self.assertEqual([identity(r.db.smiles[cid])[1] for cid in ids],[ethanol])
            self.assertNotIn(ether,info['oracle_keys'])
    def test_coconut_prior_preserves_candidates_and_combines_stereoisomers(self):
        candidates = pd.DataFrame({'key': ['ABCDEFGHIJKLMN', 'UNMATCHEDKEYXX'],
                                   'source': ['train', 'train'], 'mass': [100., 200.]})
        raw = pd.DataFrame({'standard_inchi_key': ['ABCDEFGHIJKLMN-A', 'ABCDEFGHIJKLMN-B', None],
                            'annotation_level': [2, 4, 5]})
        result = enrich(candidates, raw)
        pd.testing.assert_frame_equal(result[candidates.columns], candidates)
        self.assertEqual(result.coconut_records.tolist(), [2, 0])
        self.assertEqual(result.coconut_annotation.tolist(), [4., 0.])
    def test_adducts(self):
        mass=formula_mass('C6H12O6')
        for a in ['[M+H]+','[M-H]-','[M+Na]+','[M+K]+','[M+NH4]+','[M+H-H2O]+','[M-H2O+H]+','[M+CH2O2-H]-','[M+Cl]-','[2M+Na]+','[M+2H]2+']:
            n,z,s=adduct_parameters(a);self.assertAlmostEqual(float(neutral_mass((n*mass+s)/abs(z),a)),mass,places=8)
        self.assertAlmostEqual(float(neutral_mass(181.07066457,'[M+H]+')),180.0633881,places=6)
        self.assertAlmostEqual(float(neutral_mass(179.05611165,'[M-H]-')),180.0633881,places=6)
        with self.assertRaises(ValueError):neutral_mass(100,'[unknown]+')
    def test_equivalence(self):
        self.assertEqual(identity('CC(=O)C')[1],identity('C=C(O)C')[1])
        self.assertEqual(identity('C[C@H](O)C(=O)O')[1],identity('C[C@@H](O)C(=O)O')[1])
        self.assertEqual(len(deduplicate(['CCO','OCC','CCC'])),2)
    def test_metric(self):
        self.assertEqual(evaluate(['CCO'],[['invalid','CCO']])['MRR25'],.5)
        self.assertEqual(evaluate(['CCO'],[['CCC']*24+['CCO']])['MRR25'],.04)
        self.assertEqual(evaluate(['CCO'],[['CCC']*25+['CCO']])['MRR25'],0)
    def test_submission(self):
        good=pd.DataFrame({'molecule_id':['b','a'],'smiles':['CCO;CCC','CCN']});check_submission(good,['b','a'])
        for bad in [good.iloc[::-1],pd.concat([good,good]),good.assign(smiles=';'.join(['CCO']*26)),good.assign(smiles='CCO;OCC'),good.assign(smiles='invalid')]:
            with self.assertRaises(ValueError):check_submission(bad,['b','a'])
    def test_groups(self):
        d=pd.DataFrame({'key':['a','b','c'],'scaffold':['','', 'C1CC1']});s=assignments(d,'scaffold')
        self.assertEqual(s.fold.iloc[0],s.fold.iloc[1])
        with self.assertRaises(AssertionError):assert_disjoint(['a'],['a'])
    def test_matching(self):
        q=clean_peaks([10,20],[1,1]);r=clean_peaks([11,21],[1,1])
        self.assertAlmostEqual(modified_cosine(q,r,-1),1,places=6)
        x=clean_peaks([10,10],[1,2]);self.assertEqual(np.count_nonzero(x[1]),1)
    def test_candidate_db_deduplicates_and_reports_coverage(self):
        frame=pd.DataFrame({'mass':[101.,100.,100.], 'key':['b','a','a'], 'smiles':['CC','C','C']})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'c.parquet';frame.to_parquet(path,index=False)
            db=CandidateDB(path)
            self.assertEqual(db.key.tolist(),['a','b'])
            ids,_,fallback=db.retrieve([100.0005],ppm=10,minimum_da=0)
            self.assertFalse(fallback);self.assertEqual(db.key[ids].tolist(),['a'])
            self.assertTrue(db.coverage([100.0005],'a',ppm=10,minimum_da=0))
            self.assertFalse(db.coverage([100.0005],'b',ppm=10,minimum_da=0))
if __name__=='__main__':unittest.main()
