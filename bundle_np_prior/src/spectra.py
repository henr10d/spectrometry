import numpy as np
from scipy import sparse

def clean_peaks(mz,intensity,top_k=64):
    mz=np.asarray(mz,dtype=np.float64); y=np.asarray(intensity,dtype=np.float64)
    if mz.shape!=y.shape: raise ValueError('Peak/intensity length mismatch')
    good=np.isfinite(mz)&np.isfinite(y)&(mz>0)&(y>0)
    mz,y=mz[good],y[good]
    # Sum repeated exact masses before normalization and top-k selection.
    mz,inv=np.unique(mz,return_inverse=True);y=np.bincount(inv,weights=y)
    if len(y)>top_k:
        keep=np.argsort(-y,kind='stable')[:top_k];mz,y=mz[keep],y[keep]
    order=np.argsort(mz);mz,y=mz[order],y[order]
    out=np.zeros((2,top_k),dtype=np.float32)
    out[0,:len(mz)]=mz
    if len(y): out[1,:len(y)]=y/y.max()
    return out

def binned(mz,intensity,precursor=None,width=.1,max_mz=2000):
    """Sparse square-root intensity vectors. Loss bins assume singly charged ions."""
    mz=np.atleast_2d(mz);y=np.atleast_2d(intensity)
    if precursor is not None: mz=np.asarray(precursor).reshape(-1,1)-mz
    valid=np.isfinite(mz)&np.isfinite(y)&(mz>=0)&(mz<max_mz)&(y>0)
    r,c=np.where(valid);idx=np.floor(mz[r,c]/width+.5).astype(int)
    n=int(max_mz/width)+1
    mat=sparse.csr_matrix((np.sqrt(y[r,c]),(r,idx)),shape=(len(mz),n),dtype=np.float32)
    norm=np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel());norm[norm==0]=1
    return sparse.diags(1/norm).dot(mat).tocsr()

def cosine(query,library,width=.1,query_precursor=None,library_precursor=None):
    q=binned(query[0],query[1],query_precursor,width)
    l=binned(library[:,0],library[:,1],library_precursor,width)
    return (l@q.T).toarray().ravel()

def modified_cosine(query,reference,shift,tolerance=.02):
    """Greedy one-to-one peak assignment, direct or precursor-shifted matches."""
    qm,qi=query[:,query[1]>0];rm,ri=reference[:,reference[1]>0]
    qi=np.sqrt(qi);ri=np.sqrt(ri)
    edges=[]
    for i,m in enumerate(qm):
        hits=np.flatnonzero((np.abs(m-rm)<=tolerance)|(np.abs(m-rm-shift)<=tolerance))
        edges.extend((float(qi[i]*ri[j]),i,int(j)) for j in hits)
    usedq=set();usedr=set();score=0.
    for w,i,j in sorted(edges,reverse=True):
        if i not in usedq and j not in usedr: score+=w;usedq.add(i);usedr.add(j)
    return score/max(float(np.linalg.norm(qi)*np.linalg.norm(ri)),1e-12)
