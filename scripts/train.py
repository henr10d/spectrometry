"""Benchmark a lightweight learned baseline on the same frozen validation queries.
Candidate universe is closed and explicit; spectra from held structures never train.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.inference import Retriever
from src.validation import assignments
from src.chemistry import neutral_mass
from src.spectra import clean_peaks
from src.models.fingerprint_model import features,FingerprintRidge,soft_tanimoto
from src.metric import evaluate_keys

def fit_model(r,held,n,seed=2026):
    start=time.time();rng=np.random.default_rng(seed)
    good=(~r.meta.key.isin(held))&(r.ids>=0)&(r.meta.instrument_type=='timsTOF')&r.meta.precursor_mz.between(100,1000)&r.meta.adduct.isin(['[M+H]+','[M-H]-'])
    # One random spectrum per structure prevents abundant structures dominating training.
    eligible=r.meta[good].sample(frac=1,random_state=seed).drop_duplicates('key')
    rows=eligible.sample(n=min(n,len(eligible)),random_state=seed).index.to_numpy()
    assert not (set(r.meta.key.iloc[rows])&set(held))
    x=features(r.peaks[rows],r.precursor[rows],r.mode[rows]=='positive')
    fp=np.unpackbits(np.stack([np.frombuffer(x,dtype=np.uint8) for x in r.db.table.fingerprint]),axis=1).astype(np.float32)
    y=fp[r.ids[rows]];m=FingerprintRidge.fit(x,y)
    return m,fp,time.time()-start,len(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n-train',type=int,default=40000);ap.add_argument('--final',action='store_true');a=ap.parse_args()
    r=Retriever('artifacts',json.load(open('configs/baseline.json')))
    if a.final:
        m,fp,sec,n=fit_model(r,set(),a.n_train);m.save('artifacts/fingerprint_ridge.npz')
        Path('notes/fingerprint_training.json').write_text(json.dumps({'seconds':sec,'training_structures':n,'seed':2026,'alpha':20,'features':'2 Da fragment/loss bins plus precursor, mode, intercept'},indent=2));print(n,sec);return
    detail=pd.read_csv('experiments/query_results.csv');result=pd.read_csv('experiments/results.csv');added=[];query_rows=[]
    for kind in ['random','scaffold','np_domain']:
        keys=sorted(detail.loc[(detail.split==kind)&(detail.protocol=='closed_universe')].key.unique())
        if kind=='np_domain':held=set(r.meta.loc[r.meta.ingest_lib=='enveda-np-examples','key'].dropna())
        else:
            split=assignments(r.db.table,kind);held=set(split.loc[split.fold=='validation','key'])
        m,fp,training_time,n=fit_model(r,held,a.n_train);m.save(f'artifacts/fingerprint_{kind}.npz')
        preds=[];hybrid_preds=[];oracles=[];start=time.time();r.fingerprint_model=m
        for key in keys:
            good=(r.meta.key==key)&(r.meta.instrument_type=='timsTOF')&r.meta.precursor_mz.between(100,1000)
            if kind=='np_domain':good &= r.meta.ingest_lib=='enveda-np-examples'
            rows=r.meta[good].index.to_numpy();rows=rows[np.argsort([str(r.adduct[x]) for x in rows],kind='stable')][:9]
            probs=m.predict(features(r.peaks[rows],r.precursor[rows],r.mode[rows]=='positive')).mean(axis=0)
            masses=[float(neutral_mass(r.precursor[x],r.adduct[x])) for x in rows]
            ids,err,fallback=r.db.retrieve(masses,ppm=r.config['mass_ppm'],minimum_da=r.config['mass_floor_da'])
            q=r.meta.iloc[rows].copy();q['ms2_mzs']=[r.peaks[x,0] for x in rows];q['ms2_normalized_intensities']=[r.peaks[x,1] for x in rows]
            hybrid_ids,_=r.predict(q,method='hybrid',excluded_keys=held);hybrid_preds.append(r.db.key[hybrid_ids].tolist())
            score=soft_tanimoto(probs,fp[ids]);order=np.lexsort((r.db.key[ids],err,-score));p=r.db.key[ids[order[:25]]].tolist();preds.append(p);oracles.append(key in r.db.key[ids])
            query_rows.append({'split':kind,'key':key,'rank':next((i+1 for i,k in enumerate(p) if k==key),None),'oracle':oracles[-1]})
        score=evaluate_keys(keys,preds);rec={'experiment':'fingerprint_ridge_closed_universe','validation_split':kind,**score,'candidate_oracle_recall':np.mean(oracles),'training_time':training_time,'inference_time':time.time()-start,'notes':f'n={len(keys)}; train={n}; strict held identities; closed-universe structures; 2Da bins; alpha20; no tuning'}
        added.append(rec);print(rec,flush=True)
        hybrid_rec={**rec,'experiment':'hybrid_closed_universe',**evaluate_keys(keys,hybrid_preds)};added.append(hybrid_rec);print(hybrid_rec,flush=True)
        pd.concat([result,pd.DataFrame(added)],ignore_index=True).to_csv('experiments/results.csv',index=False)
        pd.DataFrame(query_rows).to_csv('experiments/fingerprint_query_results.csv',index=False)
if __name__=='__main__':main()
