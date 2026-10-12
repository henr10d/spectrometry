"""Read metadata without loading the 3 GB peak arrays into memory."""
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

def main():
    Path('notes').mkdir(exist_ok=True); Path('artifacts').mkdir(exist_ok=True)
    out={}
    for name in ['train','test']:
        p=pq.ParquetFile(f'{name}.parquet')
        cols=[c for c in p.schema_arrow.names if c not in ['ms2_mzs','ms2_normalized_intensities','collision_energy_ev']]
        d=p.read(columns=cols).to_pandas()
        out[name]={'rows':len(d),'bytes':Path(f'{name}.parquet').stat().st_size,'schema':str(p.schema_arrow),'nulls':d.isna().sum().to_dict()}
        for c in ['ingest_lib','adduct','instrument_type','ionization_mode','collision_energy_orig_units']:
            if c in d: out[name][c]=d[c].fillna('<NULL>').value_counts().to_dict()
        for c in ['precursor_mz','precursor_error_ppm','num_peaks']:
            if c in d: out[name][c]=d[c].describe(percentiles=[.01,.5,.95,.99]).to_dict()
        key='inchikey14' if name=='train' else 'molecule_id'
        out[name]['unique_molecules']=d[key].nunique()
        out[name]['spectra_per_molecule']=d.groupby(key).size().describe().to_dict()
        if name=='train':
            out[name]['unique_smiles']=d.normalized_smiles.nunique()
            out[name]['unique_formula']=d.molecular_formula.nunique()
            out[name]['libraries']=d.groupby('ingest_lib').agg(spectra=('inchikey14','size'),structures=('inchikey14','nunique')).to_dict('index')
            out[name]['structures_in_multiple_libraries']=int((d.groupby(key).ingest_lib.nunique()>1).sum())
            d.to_parquet('artifacts/train_metadata.parquet',index=False)
    Path('notes/data_statistics.json').write_text(json.dumps(out,indent=2,default=int))
    print('Wrote notes/data_statistics.json')
if __name__=='__main__': main()
