import numpy as np
from .chemistry import identity

def evaluate_keys(truth,predictions):
    if len(truth)!=len(predictions) or not len(truth): raise ValueError('Mismatched/empty inputs')
    ranks=[]
    for key,p in zip(truth,predictions):
        ranks.append(next((i+1 for i,x in enumerate(p[:25]) if x==key),np.inf))
    r=np.array(ranks)
    return {'MRR25':float(np.mean(1/r)),**{f'R{k}':float(np.mean(r<=k)) for k in [1,5,10,25]}}

def evaluate(truth,predictions):
    # Invalid predictions occupy their original rank; never silently compress ranks.
    def key(s):
        try: return identity(s)[1]
        except (ValueError,RuntimeError): return None
    return evaluate_keys([identity(s)[1] for s in truth],[[key(s) for s in p] for p in predictions])
