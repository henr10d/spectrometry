import json
import pandas as pd
import numpy as np
from src.inference import Retriever
from src.chemistry import neutral_mass
from src.validation import stable_uniform
from scripts.validate_np_holdout import build_queries
r=Retriever('artifacts',json.load(open('configs/ensemble.json')))
raw=pd.read_csv('external/coconut_csv_lite-10-2026.csv',usecols=['standard_inchi_key','annotation_level','name','np_likeness'],low_memory=False)
raw['key']=raw.standard_inchi_key.str[:14];raw['named']=raw.name.notna().astype(float)
p=raw.groupby('key').agg(annotation=('annotation_level','max'),named=('named','max'),count=('key','size'),np=('np_likeness','max'))
keys=sorted(r.meta.loc[r.meta.ingest_lib.eq('enveda-np-examples')&r.meta.instrument_type.eq('timsTOF')&r.meta.precursor_mz.between(100,1000),'key'].dropna().unique())
queries=build_queries(r,keys);out=[]
for key,q in zip(keys,queries):
 ids,err,_=r.db.retrieve([float(neutral_mass(z.precursor_mz,z.adduct)) for z in q.itertuples()]);v=p.reindex(r.db.key[ids]).fillna(0)
 scores=[np.zeros(len(ids)),v.annotation.to_numpy(),v.named.to_numpy(),np.log1p(v['count'].to_numpy()),v.annotation.to_numpy()+v.named.to_numpy(),v.annotation.to_numpy()+.5*np.log1p(v['count'].to_numpy())]
 ranks=[]
 for s in scores:
  order=np.lexsort((r.db.key[ids],err,-s));hits=np.flatnonzero(r.db.key[ids[order]]==key);ranks.append(int(hits[0]+1) if len(hits) else 0)
 out.append({'key':key,'ranks':ranks})
rr=np.array([[1/z if 0<z<=25 else 0 for z in t['ranks']] for t in out]);dev=np.array([stable_uniform(k,42)<.5 for k in keys]);print('DEV',rr[dev].mean(0),'HOLD',rr[~dev].mean(0),'ALL',rr.mean(0),flush=True)
open('experiments/prior_probe.json','w').write(json.dumps(out,indent=2))
