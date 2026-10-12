"""Max Morgan similarity to every allowed training structure, not just a random sample."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import DataStructs
from src.validation import assignments

def main():
    c=pd.read_parquet('artifacts/candidates.parquet');s=pd.read_parquet('artifacts/structure_map.parquet');m=pd.read_parquet('artifacts/train_metadata.parquet',columns=['normalized_smiles','ingest_lib'])
    maps=dict(zip(s.normalized_smiles,s.key));m['key']=m.normalized_smiles.map(maps)
    bits=[DataStructs.CreateFromBinaryText(x) for x in c.fingerprint];lookup={k:i for i,k in enumerate(c.key)}
    queries=pd.read_csv('experiments/query_results.csv');distances=[]
    for split,group in queries.groupby('split'):
        if split=='np_domain':held=set(m.loc[m.ingest_lib=='enveda-np-examples','key'].dropna())
        else:
            folds=assignments(c,split);held=set(folds.loc[folds.fold=='validation','key'])
        refs=[bits[i] for i,k in enumerate(c.key) if k not in held]
        for key in sorted(group.key.unique()):
            sim=max(DataStructs.BulkTanimotoSimilarity(bits[lookup[key]],refs),default=0.)
            distances.append({'split':split,'key':key,'max_train_tanimoto':sim,'stratum':'close_analog_ge_0.7' if sim>=.7 else ('distant_le_0.4' if sim<=.4 else 'intermediate')})
    dist=pd.DataFrame(distances);dist.to_csv('experiments/novelty.csv',index=False)
    if Path('experiments/fingerprint_query_results.csv').exists():
        f=pd.read_csv('experiments/fingerprint_query_results.csv');f['method']='fingerprint_ridge';f['protocol']='closed_universe';queries=pd.concat([queries,f],ignore_index=True)
    q=queries.merge(dist,on=['split','key'],validate='many_to_one');out=[]
    for (split,protocol,method,stratum),g in q.groupby(['split','protocol','method','stratum']):
        rank=g['rank'].fillna(np.inf);out.append({'split':split,'protocol':protocol,'method':method,'stratum':stratum,'n':len(g),'MRR25':(1/rank).mean(),**{f'R{k}':(rank<=k).mean() for k in [1,5,10,25]},'candidate_oracle_recall':g.oracle.mean()})
    pd.DataFrame(out).to_csv('experiments/stratified_results.csv',index=False);print(dist.groupby(['split','stratum']).size().to_string())
if __name__=='__main__':main()
