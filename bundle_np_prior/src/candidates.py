from pathlib import Path
import numpy as np
import pandas as pd

class CandidateDB:
    def __init__(self,path):
        path=Path(path)
        self.table=pd.read_parquet(path)
        required={'mass','key','smiles'}
        missing=required-set(self.table.columns)
        if missing:raise ValueError(f'Candidate table is missing columns: {sorted(missing)}')
        self.table=self.table.dropna(subset=list(required)).sort_values(['mass','key'],kind='stable').drop_duplicates('key',keep='first').reset_index(drop=True)
        if not len(self.table):raise ValueError('Candidate table is empty')
        self.mass=self.table.mass.to_numpy();self.key=self.table.key.to_numpy();self.smiles=self.table.smiles.to_numpy()
        self.key_to_id={k:i for i,k in enumerate(self.key)}
        self.pop=self.table['pop'].to_numpy(dtype=float) if 'pop' in self.table.columns else np.zeros(len(self.table))
    def retrieve(self,masses,ppm=10,minimum_da=.002):
        masses=np.array(masses,dtype=float);masses=masses[np.isfinite(masses)&(masses>0)]
        if not len(masses): raise ValueError('No valid neutral masses')
        # Union of windows tolerates one bad adduct; robust median ranks the union.
        ids=[]
        for m in masses:
            t=max(m*ppm*1e-6,minimum_da)
            ids.extend(range(np.searchsorted(self.mass,m-t),np.searchsorted(self.mass,m+t,side='right')))
        fallback=False
        if not ids:
            # Explicit low-confidence escape hatch to guarantee nonempty legal output.
            m=np.median(masses);j=np.searchsorted(self.mass,m)
            ids=list(range(max(0,j-25),min(len(self.mass),j+25)));fallback=True
        ids=np.unique(ids)
        error=np.abs(self.mass[ids]-np.median(masses))/np.median(masses)*1e6
        order=np.lexsort((self.key[ids],error))
        return ids[order],error[order],fallback

    def coverage(self,masses,key,ppm=10,minimum_da=.002):
        """Return mass-window membership without invoking fallback retrieval."""
        masses=np.asarray(masses,dtype=float)
        masses=masses[np.isfinite(masses)&(masses>0)]
        idx=self.key_to_id.get(key)
        if idx is None or not len(masses):return False
        return bool(np.any(np.abs(self.mass[idx]-masses)<=np.maximum(masses*ppm*1e-6,minimum_da)))
