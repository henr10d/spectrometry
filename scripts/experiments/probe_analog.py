import time,json
import numpy as np
from scipy import sparse
from src.inference import Retriever
from src.spectra import binned
from src.chemistry import neutral_mass
from scripts.validate_np_holdout import build_queries
r=Retriever('artifacts',json.load(open('configs/ensemble.json')))
keys=sorted(r.meta.loc[r.meta.ingest_lib.eq('enveda-np-examples') & r.meta.instrument_type.eq('timsTOF') & r.meta.precursor_mz.between(100,1000),'key'].dropna().unique())
held=set(r.meta.loc[r.meta.ingest_lib.eq('enveda-np-examples'),'key'].dropna())
good=(r.ids>=0)&~r.meta.key.isin(held)&r.meta.instrument_type.eq('timsTOF')&r.meta.adduct.isin(['[M+H]+','[M-H]-'])&r.meta.precursor_mz.between(100,1000)
rows=r.meta[good].groupby(['key','adduct'],sort=False).head(3).index.to_numpy()
print('rows',len(rows),flush=True)
p=r.peaks[rows]
a=binned(p[:,0],p[:,1],width=.1,max_mz=1000);b=binned(p[:,0],p[:,1],r.precursor[rows],width=.1,max_mz=1000)
x=sparse.hstack([a,b],format='csr')*.5
bits=np.unpackbits(np.stack([np.frombuffer(z,dtype=np.uint8) for z in r.db.table.fingerprint]),axis=1)
queries=build_queries(r,keys);out=[];start=time.time()
for i,(key,q) in enumerate(zip(keys,queries)):
    masses=[float(neutral_mass(z.precursor_mz,z.adduct)) for z in q.itertuples()]
    ids,err,_=r.db.retrieve(masses)
    preds=[]
    for z in q.itertuples():
        qp=np.stack([z.ms2_mzs,z.ms2_normalized_intensities])
        v=sparse.hstack([binned(qp[0],qp[1],width=.1,max_mz=1000),binned(qp[0],qp[1],z.precursor_mz,width=.1,max_mz=1000)],format='csr')
        s=(x@v.T).toarray().ravel();s[r.mode[rows]!=z.ionization_mode]=0
        top=np.argpartition(s,-100)[-100:];top=top[np.argsort(-s[top])]
        nb=r.ids[rows[top]];cb=bits[ids].astype(np.float32);nbits=bits[nb].astype(np.float32)
        dot=cb@nbits.T;sim=dot/np.maximum(cb.sum(1)[:,None]+nbits.sum(1)[None,:]-dot,1)
        preds.append([((sim[:,:k]*s[top[:k]]**power).sum(1)/(sum(s[top[:k]]**power)+1e-8)) for k,power in [(1,1),(5,4),(20,4),(100,4),(100,8)]])
    scores=np.mean(preds,axis=0)
    ranks=[]
    for score in [np.zeros(len(ids)),*scores]:
        order=np.lexsort((r.db.key[ids],err,-score));hits=np.flatnonzero(r.db.key[ids[order]]==key);rank=int(hits[0]+1) if len(hits) else 0;ranks.append(rank)
    out.append({'key':key,'ranks':ranks})
    if i%25==0:print(i,round(time.time()-start,1),np.mean([[1/z if 0<z<=25 else 0 for z in t['ranks']] for t in out],0),flush=True)
np.savez_compressed('/tmp/analog_index.npz',rows=rows)
open('experiments/analog_probe.json','w').write(json.dumps(out,indent=2))
print('FINAL',np.mean([[1/z if 0<z<=25 else 0 for z in t['ranks']] for t in out],0),flush=True)
