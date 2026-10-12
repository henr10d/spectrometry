import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .candidates import CandidateDB
from .chemistry import neutral_mass,identity,adduct_parameters
from .spectra import clean_peaks,cosine,modified_cosine

class Retriever:
    def __init__(self,artifacts,config=None):
        self.root=Path(artifacts);self.config=config or {}
        cand=self.config.get('candidates_file','candidates.parquet')
        self.db=CandidateDB(self.root/cand)
        self.meta=pd.read_parquet(self.root/'train_metadata.parquet',columns=['normalized_smiles','adduct','precursor_mz','ionization_mode','ingest_lib','instrument_type'])
        smap=pd.read_parquet(self.root/'structure_map.parquet',columns=['normalized_smiles','key']).dropna()
        self.meta['key']=self.meta.normalized_smiles.map(dict(zip(smap.normalized_smiles,smap.key)))
        self.ids=self.meta.key.map(self.db.key_to_id).fillna(-1).to_numpy(dtype=np.int32)
        self.order=np.argsort(self.ids,kind='stable');self.sorted_ids=self.ids[self.order]
        self.peaks=np.load(self.root/'library_peaks.npy',mmap_mode='r')
        if len(self.meta)!=len(self.peaks):raise ValueError('Library/metadata length mismatch')
        self.adduct=self.meta.adduct.to_numpy();self.precursor=self.meta.precursor_mz.to_numpy();self.mode=self.meta.ionization_mode.to_numpy()
        self.charges={a:adduct_parameters(a)[1] for a in self.meta.adduct.unique() if a!='[Cat]2+'}
        self._library_candidate_ids=set(np.unique(self.sorted_ids[self.sorted_ids>=0]))
    def _library_channel(self,query,ids,excluded_keys,excluded_sources):
        rows=self.references(ids,excluded_keys,excluded_sources);positions={i:j for j,i in enumerate(ids)};per_query=[]
        for r in query.itertuples():
            scores=np.zeros(len(ids),dtype=float)
            compatible=rows[(self.mode[rows]==r.ionization_mode)&(self.adduct[rows]==r.adduct)]
            if len(compatible):
                q=clean_peaks(r.ms2_mzs,r.ms2_normalized_intensities);p=self.peaks[compatible]
                vals=np.clip(cosine(q,p,width=self.config.get('bin_width',.1)),0,1)
                dest=np.array([positions[i] for i in self.ids[compatible]]);np.maximum.at(scores,dest,vals)
            per_query.append(scores)
        if self.config.get('aggregation','mean')=='max':score=np.max(per_query,axis=0)
        else:score=np.mean(per_query,axis=0)
        return score,float(score.max())
    def _fingerprint_channel(self,query,ids):
        from .models.fingerprint_model import FingerprintRidge,features,soft_tanimoto
        if not hasattr(self,'fingerprint_model'):
            self.fingerprint_model=FingerprintRidge.load(self.root/'fingerprint_ridge.npz')
        peaks=np.stack([clean_peaks(x.ms2_mzs,x.ms2_normalized_intensities) for x in query.itertuples()])
        probs=self.fingerprint_model.predict(features(peaks,query.precursor_mz.to_numpy(),query.ionization_mode.to_numpy()=='positive')).mean(axis=0)
        bits=np.unpackbits(np.stack([np.frombuffer(x,dtype=np.uint8) for x in self.db.table.fingerprint.iloc[ids]]),axis=1).astype(np.float32)
        return soft_tanimoto(probs,bits)
    def references(self,candidate_ids,excluded_keys=(),excluded_sources=()):
        # Match dtype and search all IDs together: scalar int64 searches can copy the
        # entire 2.5M-row int32 array for each candidate.
        ids=np.asarray(candidate_ids,dtype=self.sorted_ids.dtype)
        starts=np.searchsorted(self.sorted_ids,ids)
        ends=np.searchsorted(self.sorted_ids,ids,side='right')
        blocks=[self.order[a:b] for a,b in zip(starts,ends)]
        rows=np.concatenate(blocks) if blocks else np.array([],dtype=int)
        keep=np.ones(len(rows),bool)
        if len(excluded_keys):keep &= ~self.meta.key.iloc[rows].isin(excluded_keys).to_numpy()
        if len(excluded_sources):keep &= ~self.meta.ingest_lib.iloc[rows].isin(excluded_sources).to_numpy()
        return rows[keep]
    def predict(self,query,method='cosine',excluded_keys=(),excluded_sources=(),candidate_excluded_keys=(),canonical_oracle=False):
        if method not in ['mass','cosine','loss','modified','fingerprint','hybrid','gated_mass','ensemble','np_prior']:raise ValueError(f'Unknown method {method}')
        masses=[float(neutral_mass(r.precursor_mz,r.adduct)) for r in query.itertuples()]
        ids,errors,fallback=self.db.retrieve(masses,ppm=self.config.get('mass_ppm',10),minimum_da=self.config.get('mass_floor_da',.002))
        oracle_keys=(np.array([identity(self.db.smiles[cid])[1] for cid in ids])
                     if canonical_oracle or len(candidate_excluded_keys) else self.db.key[ids])
        if len(candidate_excluded_keys):
            mask=~np.isin(oracle_keys,list(candidate_excluded_keys))
            ids,errors,oracle_keys=ids[mask],errors[mask],oracle_keys[mask]
        if not len(ids): return [],{'oracle_keys':[],'fallback':fallback,'candidates':0}
        per_query=[];rows=self.references(ids,excluded_keys,excluded_sources) if method not in ['mass','fingerprint'] else np.array([],dtype=int)
        positions={i:j for j,i in enumerate(ids)}
        for r in query.itertuples():
            scores=np.zeros(len(ids),dtype=float)
            compatible=rows[(self.mode[rows]==r.ionization_mode)&(self.adduct[rows]==r.adduct)]
            if len(compatible) and method not in ['mass','fingerprint']:
                q=clean_peaks(r.ms2_mzs,r.ms2_normalized_intensities)
                p=self.peaks[compatible]
                vals=cosine(q,p,width=self.config.get('bin_width',.1))
                if method in ['loss','modified']:
                    # Neutral losses in m/z only used for matching singly charged adducts.
                    if abs(self.charges.get(r.adduct,0))==1:
                        loss=cosine(q,p,width=self.config.get('bin_width',.1),query_precursor=r.precursor_mz,library_precursor=self.precursor[compatible])
                        vals=.5*vals+.5*loss
                if method=='modified':
                    # Limit exact matching to a deterministic cosine shortlist.
                    top=np.argsort(-vals,kind='stable')[:self.config.get('modified_shortlist',100)]
                    refined=np.zeros(len(vals))
                    for j in top:refined[j]=modified_cosine(q,p[j],r.precursor_mz-self.precursor[compatible[j]])
                    vals=refined
                vals=np.clip(vals,0,1)
                dest=np.array([positions[i] for i in self.ids[compatible]])
                np.maximum.at(scores,dest,vals)
            per_query.append(scores)
        if self.config.get('aggregation','mean')=='max':score=np.max(per_query,axis=0)
        else:score=np.mean(per_query,axis=0)
        spectral_best=float(score.max())
        if method=='np_prior' and spectral_best<self.config.get('exact_match_gate',0.999):
            required={'coconut_annotation','coconut_records'}
            if not required.issubset(self.db.table.columns):
                raise ValueError('np_prior requires candidates built with enrich_coconut_prior.py')
            table=self.db.table.iloc[ids]
            score=(table.coconut_annotation.to_numpy(dtype=float)
                   +self.config.get('record_weight',.5)*np.log1p(table.coconut_records.to_numpy(dtype=float)))
        if method=='gated_mass' and spectral_best<self.config.get('exact_match_gate',0.999):score=np.zeros_like(score)
        if method=='ensemble':
            lib,spectral_best=self._library_channel(query,ids,excluded_keys,excluded_sources)
            fp=self._fingerprint_channel(query,ids)
            w=self.config.get('ensemble',{})
            gate=self.config.get('exact_match_gate',0.999)
            if spectral_best>=gate:score=lib.copy()
            else:
                err_score=1-errors/(errors.max()+1e-6)
                train=np.array([i in self._library_candidate_ids for i in ids],dtype=float)
                pop=self.db.pop[ids];pop=pop/(pop.max()+1e-6) if pop.max()>0 else pop
                score=(w.get('w_library',.55)*lib+w.get('w_fingerprint',.35)*fp+w.get('w_train_prior',.12)*train
                       +w.get('w_popularity',.2)*pop+w.get('w_mass',.08)*err_score)
        if method in ['fingerprint','hybrid']:
            from .models.fingerprint_model import FingerprintRidge,features,soft_tanimoto
            if not hasattr(self,'fingerprint_model'):
                self.fingerprint_model=FingerprintRidge.load(self.root/'fingerprint_ridge.npz')
            peaks=np.stack([clean_peaks(x.ms2_mzs,x.ms2_normalized_intensities) for x in query.itertuples()])
            probs=self.fingerprint_model.predict(features(peaks,query.precursor_mz.to_numpy(),query.ionization_mode.to_numpy()=='positive')).mean(axis=0)
            bits=np.unpackbits(np.stack([np.frombuffer(x,dtype=np.uint8) for x in self.db.table.fingerprint.iloc[ids]]),axis=1).astype(np.float32)
            fp_score=soft_tanimoto(probs,bits)
            # An exact-like spectral match is retained; otherwise use the learned ranker.
            # This threshold is fixed, and its effect is separately validated.
            if method=='fingerprint' or spectral_best<self.config.get('exact_match_gate',0.999):score=fp_score
        # Mass breaks ties, never lets an implausible outside-window candidate win.
        order=np.lexsort((self.db.key[ids],errors,-score))
        # External databases may use a different tautomer for their connectivity
        # key. Deduplicate with the competition identity before emitting SMILES.
        top=[];seen=set()
        for cid in ids[order]:
            try:key=identity(self.db.smiles[cid])[1]
            except ValueError:continue
            if key in seen:continue
            seen.add(key);top.append(int(cid))
            if len(top)==25:break
        return top,{'oracle_keys':oracle_keys.tolist(),'fallback':fallback,'candidates':len(ids),'best_score':float(score.max()),'spectral_best':spectral_best}

def check_submission(frame,expected):
    if list(frame.columns)!=['molecule_id','smiles']:raise ValueError('Wrong columns')
    if frame.isna().any().any() or frame.molecule_id.duplicated().any():raise ValueError('Null/duplicate IDs')
    if frame.molecule_id.tolist()!=list(expected):raise ValueError('Molecule IDs/order mismatch')
    counts=[]
    for s in frame.smiles:
        guesses=s.split(';')
        if not 1<=len(guesses)<=25:raise ValueError('Invalid candidate count')
        keys=[identity(g)[1] for g in guesses]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate equivalent candidates')
        counts.append(len(keys))
    return {'test_molecules':len(frame),'average_candidates':float(np.mean(counts)),'min_candidates':min(counts),'max_candidates':max(counts),'invalid_smiles':0,'duplicate_candidates':0}

def run(test,template,artifacts,output,config):
    start=time.time();r=Retriever(artifacts,config);d=pd.read_parquet(test);sample=pd.read_csv(template)
    template_matches=set(d.molecule_id)==set(sample.molecule_id)
    expected=sample.molecule_id.tolist() if template_matches else d.molecule_id.drop_duplicates().tolist()
    grouped={k:v for k,v in d.groupby('molecule_id',sort=False)};pred=[];audit=[]
    for i,k in enumerate(expected):
        ids,info=r.predict(grouped[k],method=config.get('method','cosine'))
        pred.append(';'.join(r.db.smiles[ids]));audit.append({'molecule_id':k,**{a:b for a,b in info.items() if a!='oracle_keys'}})
        if i%100==0:print('inference',i,round(time.time()-start,1),flush=True)
    frame=pd.DataFrame({'molecule_id':expected,'smiles':pred})
    stats=check_submission(frame,expected);stats['template_ids_match_test']=template_matches;stats['seconds']=time.time()-start
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(output,index=False)
    stats['output']=str(output);stats['fallback_queries']=sum(x['fallback'] for x in audit)
    output.with_suffix('.stats.json').write_text(json.dumps(stats,indent=2));pd.DataFrame(audit).to_csv(output.with_suffix('.audit.csv'),index=False)
    print(json.dumps(stats,indent=2));return stats
