"""Offline, bounded-memory preprocessing of every training spectrum."""
import time,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from src.spectra import clean_peaks

def main():
    p=pq.ParquetFile('train.parquet');n=p.metadata.num_rows;start=time.time()
    out=np.lib.format.open_memmap('artifacts/library_peaks.partial.npy',mode='w+',dtype='float32',shape=(n,2,64))
    offset=0;stats={'rows':n,'bad_lengths':0,'empty_after_cleaning':0,'top_k':64}
    for batch in p.iter_batches(batch_size=8192,columns=['ms2_mzs','ms2_normalized_intensities']):
        arrays=batch.to_pydict()
        for i,(mz,y) in enumerate(zip(arrays['ms2_mzs'],arrays['ms2_normalized_intensities'])):
            if len(mz)!=len(y):stats['bad_lengths']+=1;raise ValueError(f'Bad row {offset+i}')
            clean=clean_peaks(mz,y);out[offset+i]=clean
            stats['empty_after_cleaning']+=int(not clean[1].any())
        offset+=len(batch)
        if offset%131072==0: print(offset,n,round(time.time()-start),flush=True)
    out.flush();del out
    Path('artifacts/library_peaks.partial.npy').replace('artifacts/library_peaks.npy')
    stats['seconds']=time.time()-start
    Path('notes/library_build.json').write_text(json.dumps(stats,indent=2));print(stats)
if __name__=='__main__':main()
