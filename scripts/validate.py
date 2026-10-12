"""Strict grouped held-out evaluation with explicitly labeled candidate protocols."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.inference import Retriever
from src.validation import assignments,assert_disjoint
from src.metric import evaluate_keys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=128);ap.add_argument('--splits',nargs='+',default=['random','scaffold','np_domain']);args=ap.parse_args()
    cfg=json.load(open('configs/baseline.json'));r=Retriever('artifacts',cfg);start=time.time();rng=np.random.default_rng(cfg['seed'])
    results=[];details=[]
    for kind in args.splits:
        if kind=='np_domain':
            held=set(r.meta.loc[r.meta.ingest_lib=='enveda-np-examples','key'].dropna());sources=['enveda-np-examples']
        else:
            split=assignments(r.db.table,kind,seed=cfg['seed']);split.to_csv(f'artifacts/split_{kind}.csv',index=False)
            held=set(split.loc[split.fold=='validation','key']);sources=None
        assert_disjoint(set(r.meta.key.dropna())-held,held)
        eligible=r.meta[r.meta.key.isin(held)&(r.meta.instrument_type=='timsTOF')&(r.meta.precursor_mz.between(100,1000))]
        if sources:eligible=eligible[eligible.ingest_lib.isin(sources)]
        keys=sorted(eligible.key.unique());keys=sorted(rng.choice(keys,min(len(keys),args.n),replace=False))
        queries=[]
        for key in keys:
            rows=eligible[eligible.key==key].index.to_numpy()
            rows=rows[np.argsort([str(r.adduct[x]) for x in rows],kind='stable')][:9]
            q=r.meta.iloc[rows].copy();q['ms2_mzs']=[r.peaks[x,0] for x in rows];q['ms2_normalized_intensities']=[r.peaks[x,1] for x in rows];queries.append(q)
        for protocol in ['train_only','closed_universe']:
            for method in (['mass'] if protocol=='train_only' else ['mass','cosine','loss','modified']):
                t=time.time();pred=[];oracle=[];counts=[]
                for key,q in zip(keys,queries):
                    ids,info=r.predict(q,method=method,excluded_keys=held,candidate_excluded_keys=held if protocol=='train_only' else ())
                    pred.append(r.db.key[ids].tolist());oracle.append(key in info['oracle_keys']);counts.append(info['candidates'])
                    details.append({'split':kind,'protocol':protocol,'method':method,'key':key,'oracle':oracle[-1],'candidates':counts[-1],'rank':next((i+1 for i,x in enumerate(pred[-1]) if x==key),None)})
                score=evaluate_keys(keys,pred)
                result={'experiment':method+'_'+protocol,'validation_split':kind,**score,'candidate_oracle_recall':float(np.mean(oracle)),'training_time':0,'inference_time':time.time()-t,'notes':f'n={len(keys)}; seed=2026; timsTOF m/z 100-1000; all held identities excluded globally; closed universe includes held structures only'}
                results.append(result);print(result,flush=True)
                pd.DataFrame(results).to_csv('experiments/results.csv',index=False)
        pd.DataFrame(details).to_csv('experiments/query_results.csv',index=False)
    print('total seconds',time.time()-start)
if __name__=='__main__':main()
