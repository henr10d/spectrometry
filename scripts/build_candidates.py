import argparse,time,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import rdkit
from src.chemistry import describe

def worker(s):
    try: return {'normalized_smiles':s,**describe(s)}
    except Exception as e: return {'normalized_smiles':s,'error':str(e)}

def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=8);args=a.parse_args()
    assert rdkit.__version__=='2026.03.3', rdkit.__version__
    start=time.time();d=pd.read_parquet('artifacts/train_metadata.parquet')
    smiles=sorted(d.normalized_smiles.unique())
    with ProcessPoolExecutor(args.workers) as pool:
        rows=[]
        for i,r in enumerate(pool.map(worker,smiles,chunksize=128)):
            rows.append(r)
            if i%20000==0: print(i,len(smiles),round(time.time()-start),flush=True)
    m=pd.DataFrame(rows);m.to_parquet('artifacts/structure_map.parquet',index=False)
    valid=m[m.key.notna()].copy(); c=valid.sort_values('smiles').drop_duplicates('key').sort_values(['mass','key']).reset_index(drop=True)
    c=c.assign(source='train',pop=0.0)
    c.to_parquet('artifacts/candidates.parquet',index=False)
    Path('notes/candidate_build.json').write_text(json.dumps({'input':len(m),'valid':len(valid),'canonical_candidates':len(c),'seconds':time.time()-start,'rdkit':rdkit.__version__},indent=2))
if __name__=='__main__':main()
