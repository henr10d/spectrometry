"""Hard stress test: remove every training structure >0.4 Morgan Tanimoto
from any of 64 seeded natural-product queries, then refit the small model.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import DataStructs
from src.inference import Retriever
from src.chemistry import neutral_mass
from src.models.fingerprint_model import features,soft_tanimoto
from src.metric import evaluate_keys
from scripts.train import fit_model

def main():
    r=Retriever('artifacts',json.load(open('configs/baseline.json')));rng=np.random.default_rng(2026)
    eligible=r.meta[(r.meta.ingest_lib=='enveda-np-examples')&r.meta.precursor_mz.between(100,1000)]
    keys=sorted(rng.choice(sorted(eligible.key.unique()),64,replace=False));bits=[DataStructs.CreateFromBinaryText(x) for x in r.db.table.fingerprint]
    heldmask=np.zeros(len(bits),bool)
    for key in keys:heldmask |= np.asarray(DataStructs.BulkTanimotoSimilarity(bits[r.db.key_to_id[key]],bits))>.4
    held=set(r.db.key[heldmask]);assert set(keys)<=held
    pd.DataFrame({'key':r.db.key,'fold':np.where(heldmask,'excluded','train'),'query':np.isin(r.db.key,keys)}).to_csv('artifacts/split_novelty.csv',index=False)
    model,fp,training_time,n=fit_model(r,held,40000);model.save('artifacts/fingerprint_novelty.npz')
    preds={'mass':[],'fingerprint_ridge':[]};oracle=[];max_sims=[];refs=[b for i,b in enumerate(bits) if not heldmask[i]];start=time.time()
    for key in keys:
        q=eligible[eligible.key==key];rows=q.index.to_numpy();rows=rows[np.argsort([str(r.adduct[x]) for x in rows],kind='stable')][:9]
        ids,errors,fallback=r.db.retrieve([float(neutral_mass(r.precursor[x],r.adduct[x])) for x in rows]);oracle.append(key in r.db.key[ids]);preds['mass'].append(r.db.key[ids[:25]].tolist())
        p=model.predict(features(r.peaks[rows],r.precursor[rows],r.mode[rows]=='positive')).mean(axis=0);score=soft_tanimoto(p,fp[ids]);order=np.lexsort((r.db.key[ids],errors,-score));preds['fingerprint_ridge'].append(r.db.key[ids[order[:25]]].tolist())
        sim=max(DataStructs.BulkTanimotoSimilarity(bits[r.db.key_to_id[key]],refs));assert sim<=.4;max_sims.append(sim)
    records=[]
    for method,p in preds.items():
        records.append({'experiment':method+'_closed_universe','validation_split':'novelty_0.4',**evaluate_keys(keys,p),'candidate_oracle_recall':float(np.mean(oracle)),'training_time':training_time if method!='mass' else 0,'inference_time':time.time()-start,'notes':f'n=64; excluded {len(held)} structures with Tanimoto>0.4 to any query; train={n}; candidate structures retained; fixed alpha20'})
    old=pd.read_csv('experiments/results.csv');pd.concat([old,pd.DataFrame(records)],ignore_index=True).to_csv('experiments/results.csv',index=False)
    Path('notes/novelty_stress.json').write_text(json.dumps({'queries':64,'excluded_structures':len(held),'training_structures':n,'max_remaining_train_similarity':max(max_sims),'mean_remaining_train_similarity':float(np.mean(max_sims)),'results':records},indent=2));print(records)
if __name__=='__main__':main()
