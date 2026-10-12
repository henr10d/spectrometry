import time,json
import numpy as np
from scipy import sparse
from src.inference import Retriever
from src.spectra import binned
from src.models.fingerprint_model import FingerprintRidge,soft_tanimoto
from src.chemistry import neutral_mass
from src.validation import stable_uniform
from scripts.validate_np_holdout import build_queries
r=Retriever('artifacts',json.load(open('configs/ensemble.json')))
keys=sorted(r.meta.loc[r.meta.ingest_lib.eq('enveda-np-examples') & r.meta.instrument_type.eq('timsTOF') & r.meta.precursor_mz.between(100,1000),'key'].dropna().unique())
held=set(r.meta.loc[r.meta.ingest_lib.eq('enveda-np-examples'),'key'].dropna())
good=(r.ids>=0)&~r.meta.key.isin(held)&r.meta.instrument_type.eq('timsTOF')&r.meta.adduct.isin(['[M+H]+','[M-H]-'])&r.meta.precursor_mz.between(100,1000)
rows=r.meta[good].sample(frac=1,random_state=2026).drop_duplicates(['key','adduct']).index.to_numpy()
print('rows',len(rows),flush=True)
y=np.unpackbits(np.stack([np.frombuffer(z,dtype=np.uint8) for z in r.db.table.fingerprint.iloc[r.ids[rows]]]),axis=1).astype(np.float32)
queries=build_queries(r,keys)
cs=[]
for key,q in zip(keys,queries):
 ids,err,_=r.db.retrieve([float(neutral_mass(z.precursor_mz,z.adduct)) for z in q.itertuples()]);bits=np.unpackbits(np.stack([np.frombuffer(z,dtype=np.uint8) for z in r.db.table.fingerprint.iloc[ids]]),axis=1).astype(np.float32);cs.append((ids,err,bits))
def feat(p,pre,pos,width):
 a=binned(p[:,0],p[:,1],width=width,max_mz=1000);b=binned(p[:,0],p[:,1],pre,width=width,max_mz=1000)
 x=sparse.hstack([a,b],format='csr');return sparse.hstack([x.multiply(pos[:,None]),x.multiply(~pos[:,None]),sparse.csr_matrix(np.column_stack([np.ones(len(p)),pre/1000,pos]))],format='csr',dtype=np.float32)
result=[]
for width in [2.,1.]:
 start=time.time();x=feat(r.peaks[rows],r.precursor[rows],r.mode[rows]=='positive',width);print('features',width,x.shape,time.time()-start,flush=True)
 for alpha in [2.,20.]:
  start=time.time();m=FingerprintRidge.fit(x,y,alpha);print('fit',width,alpha,time.time()-start,flush=True)
  for scoring in ['tanimoto','cosine','bernoulli']:
   ranks=[]
   for key,q,(ids,err,bits) in zip(keys,queries,cs):
    p=np.stack([np.stack([z.ms2_mzs,z.ms2_normalized_intensities]) for z in q.itertuples()]);probs=m.predict(feat(p,q.precursor_mz.to_numpy(),q.ionization_mode.to_numpy()=='positive',width)).mean(0)
    if scoring=='tanimoto':s=soft_tanimoto(probs,bits)
    elif scoring=='cosine':s=(bits@probs)/np.sqrt(np.maximum(bits.sum(1),1))
    else:
     p=np.clip(probs,.02,.98);s=bits@(np.log(p)-np.log1p(-p))
    order=np.lexsort((r.db.key[ids],err,-s));hits=np.flatnonzero(r.db.key[ids[order]]==key);ranks.append(int(hits[0]+1) if len(hits) else 0)
   rr=np.array([1/z if 0<z<=25 else 0 for z in ranks]);dev=np.array([stable_uniform(k,42)<.5 for k in keys]);rec={'width':width,'alpha':alpha,'scoring':scoring,'dev':float(rr[dev].mean()),'holdout':float(rr[~dev].mean()),'mrr':float(rr.mean()),'ranks':ranks};result.append(rec);print({k:v for k,v in rec.items() if k!='ranks'},flush=True)
  m.save('/tmp/ridge_w'+str(width)+'_a'+str(alpha)+'.npz')
open('experiments/ridge_probe.json','w').write(json.dumps({'keys':keys,'results':result},indent=2))
