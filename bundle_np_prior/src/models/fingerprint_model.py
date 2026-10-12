"""Small CPU ridge baseline; no neural-model performance claims."""
import numpy as np
from scipy import sparse,linalg
from ..spectra import binned

def features(peaks,precursor,positive):
    # Separate fragment and loss channels; metadata is available at inference.
    frag=binned(peaks[:,0],peaks[:,1],width=2,max_mz=1000)
    loss=binned(peaks[:,0],peaks[:,1],precursor,width=2,max_mz=1000)
    meta=np.column_stack([np.ones(len(peaks)),np.asarray(precursor)/1000,np.asarray(positive,dtype=float)])
    return sparse.hstack([frag,loss,sparse.csr_matrix(meta)],format='csr',dtype=np.float32)

class FingerprintRidge:
    def __init__(self,weights):self.weights=np.asarray(weights,dtype=np.float32)
    @classmethod
    def fit(cls,x,y,alpha=20):
        gram=(x.T@x).toarray();gram.flat[::len(gram)+1]+=alpha
        return cls(linalg.solve(gram,np.asarray(x.T@y),assume_a='pos'))
    def predict(self,x):return np.clip(x@self.weights,0,1)
    def save(self,path):np.savez_compressed(path,weights=self.weights)
    @classmethod
    def load(cls,path):return cls(np.load(path)['weights'])

def soft_tanimoto(predicted,candidate_bits):
    dot=candidate_bits@predicted
    return dot/np.maximum(candidate_bits.sum(axis=1)+predicted.sum()-dot,1e-8)
