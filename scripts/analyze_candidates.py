import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.chemistry import neutral_mass

def main():
    m=pd.read_parquet('artifacts/train_metadata.parquet');s=pd.read_parquet('artifacts/structure_map.parquet');c=pd.read_parquet('artifacts/candidates.parquet')
    mapper=s.set_index('normalized_smiles');key=m.normalized_smiles.map(mapper.key);mass=m.normalized_smiles.map(mapper.mass).to_numpy()
    neutral=np.full(len(m),np.nan);unsupported=[]
    for adduct,idx in m.groupby('adduct').indices.items():
        try:neutral[idx]=neutral_mass(m.precursor_mz.iloc[idx].to_numpy(),adduct)
        except ValueError:unsupported.append({'adduct':adduct,'rows':len(idx)})
    error=np.abs(neutral-mass)/mass*1e6;m['calculated_abs_error_ppm']=error
    result={'candidates':len(c),'changed_supplied_key_rows':int((key!=m.inchikey14).sum()),'supplied_keys':m.inchikey14.nunique(),'canonical_keys':key.nunique(),'unsupported_adducts':unsupported,'salts_retained':int(c.smiles.str.contains('.',regex=False).sum()),'mass_errors':{}}
    for name,d in m.groupby('ingest_lib'):
        e=d.calculated_abs_error_ppm.dropna();result['mass_errors'][name]={'rows':len(d),'median_ppm':float(e.median()),'p95_ppm':float(e.quantile(.95)),'within_10ppm':float((e<=10).mean())}
    Path('notes/candidate_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
