"""Mass-stage recall before spectral/learned ranking; closed-universe diagnosis only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.inference import Retriever
from src.chemistry import neutral_mass

def main():
    r=Retriever('artifacts');q=pd.read_csv('experiments/query_results.csv');out=[]
    for split,g in q.groupby('split'):
        ranks=[];counts=[];fallbacks=0
        for key in sorted(g.key.unique()):
            eligible=r.meta[(r.meta.key==key)&(r.meta.instrument_type=='timsTOF')&r.meta.precursor_mz.between(100,1000)]
            if split=='np_domain':eligible=eligible[eligible.ingest_lib=='enveda-np-examples']
            rows=eligible.index.to_numpy();rows=rows[np.argsort([str(r.adduct[x]) for x in rows],kind='stable')][:9]
            ids,errors,fallback=r.db.retrieve([float(neutral_mass(r.precursor[x],r.adduct[x])) for x in rows]);fallbacks+=fallback
            keys=r.db.key[ids].tolist();ranks.append(keys.index(key)+1 if key in keys else np.inf);counts.append(len(ids))
        ranks=np.array(ranks)
        out.append({'split':split,'n':len(ranks),'protocol':'closed_universe_structure_only','oracle_recall':float(np.isfinite(ranks).mean()),**{f'mass_order_recall_{k}':float((ranks<=k).mean()) for k in [25,100,1000]},'candidate_count_median':float(np.median(counts)),'candidate_count_max':max(counts),'fallback_queries':fallbacks})
    Path('notes/candidate_coverage.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
