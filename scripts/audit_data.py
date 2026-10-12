"""Independent full-precision test audit and streaming exact train/test peak hash audit."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.chemistry import neutral_mass

def signature(mz,y):
    return hashlib.sha256(np.asarray(mz,dtype='<f8').tobytes()+np.asarray(y,dtype='<f8').tobytes()).digest()

def main():
    start=time.time();test=pd.read_parquet('test.parquet');out={}
    test['neutral_mass']=[float(neutral_mass(r.precursor_mz,r.adduct)) for r in test.itertuples()]
    spans=test.groupby('molecule_id').neutral_mass.agg(['min','max','median']);spans['span_ppm']=(spans['max']-spans['min'])/spans['median']*1e6
    spans.to_csv('notes/test_neutral_masses.csv');out['test_neutral_mass']=spans['median'].describe().to_dict();out['test_mass_span_ppm']=spans.span_ppm.describe(percentiles=[.5,.95,.99]).to_dict()
    out['test_peak_counts']=test.ms2_mzs.map(len).describe().to_dict();out['test_peaks_above_precursor']=int(sum(np.count_nonzero(np.asarray(r.ms2_mzs)>r.precursor_mz+.02) for r in test.itertuples()))
    out['test_negative_converted_energy']=int(sum(np.any(np.asarray(x)<0) for x in test.collision_energy_ev))
    out['test_repeated_exact_peak_mass_rows']=int(sum(len(np.unique(x))<len(x) for x in test.ms2_mzs))
    hashes={signature(r.ms2_mzs,r.ms2_normalized_intensities) for r in test.itertuples()};out['test_duplicate_peak_arrays']=len(test)-len(hashes)
    matches=[];mismatches=0;empty=0;bad=0;duplicate_mz=0;offset=0
    for b in pq.ParquetFile('train.parquet').iter_batches(batch_size=4096,columns=['ms2_mzs','ms2_normalized_intensities','inchikey14','ingest_lib']):
        d=b.to_pydict()
        for i,(mz,y) in enumerate(zip(d['ms2_mzs'],d['ms2_normalized_intensities'])):
            mz=np.asarray(mz);y=np.asarray(y)
            mismatches+=len(mz)!=len(y);empty+=len(mz)==0;bad+=int((~np.isfinite(mz)).sum()+(~np.isfinite(y)).sum())
            if len(mz)>1:duplicate_mz+=int(len(np.unique(mz))<len(mz))
            if signature(mz,y) in hashes:matches.append({'row':offset+i,'key':d['inchikey14'][i],'source':d['ingest_lib'][i]})
        offset+=len(b)
    out.update(train_test_exact_peak_matches=matches,train_bad_lengths=mismatches,train_empty_spectra=empty,train_nonfinite_peak_values=bad,train_rows_repeated_mz=duplicate_mz,seconds=time.time()-start)
    Path('notes/quality_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
