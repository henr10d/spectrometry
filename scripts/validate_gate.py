import json,time
import numpy as np
import pandas as pd
from src.inference import Retriever
from src.validation import assignments
from src.metric import evaluate_keys

def main():
    r=Retriever('artifacts',json.load(open('configs/baseline.json')));detail=pd.read_csv('experiments/query_results.csv');new=[]
    for kind in ['random','scaffold','np_domain']:
        keys=sorted(detail.loc[detail.split==kind].key.unique())
        if kind=='np_domain':held=set(r.meta.loc[r.meta.ingest_lib=='enveda-np-examples','key'].dropna())
        else:
            s=assignments(r.db.table,kind);held=set(s.loc[s.fold=='validation','key'])
        eligible=r.meta[r.meta.key.isin(keys)&(r.meta.instrument_type=='timsTOF')&r.meta.precursor_mz.between(100,1000)]
        if kind=='np_domain':eligible=eligible[eligible.ingest_lib=='enveda-np-examples']
        pred=[];oracle=[];start=time.time()
        for key in keys:
            rows=eligible[eligible.key==key].index.to_numpy();rows=rows[np.argsort([str(r.adduct[x]) for x in rows],kind='stable')][:9]
            q=r.meta.iloc[rows].copy();q['ms2_mzs']=[r.peaks[x,0] for x in rows];q['ms2_normalized_intensities']=[r.peaks[x,1] for x in rows]
            ids,info=r.predict(q,'gated_mass',excluded_keys=held);pred.append(r.db.key[ids].tolist());oracle.append(key in info['oracle_keys'])
        new.append({'experiment':'gated_mass_closed_universe','validation_split':kind,**evaluate_keys(keys,pred),'candidate_oracle_recall':np.mean(oracle),'training_time':0,'inference_time':time.time()-start,'notes':'n=128; fixed 0.999 spectral gate; otherwise mass ordering; closed-universe candidates; held structures excluded globally'})
    old=pd.read_csv('experiments/results.csv');pd.concat([old,pd.DataFrame(new)],ignore_index=True).to_csv('experiments/results.csv',index=False);print(new)
if __name__=='__main__':main()
