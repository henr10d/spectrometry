import hashlib
import numpy as np
import pandas as pd

def stable_uniform(value,seed=2026):
    return int.from_bytes(hashlib.sha256(f'{seed}:{value}'.encode()).digest()[:8],'big')/2**64

def assignments(candidates,kind='random',fraction=.2,seed=2026):
    if kind not in ['random','scaffold']:raise ValueError(kind)
    groups=candidates.key if kind=='random' else candidates.scaffold.fillna('')
    # Acyclic molecules share one group; never split an empty scaffold by identity.
    val=np.array([stable_uniform(g,seed)<fraction for g in groups])
    return pd.DataFrame({'key':candidates.key,'group':groups,'fold':np.where(val,'validation','train')})

def assert_disjoint(train_keys,val_keys):
    overlap=set(train_keys)&set(val_keys)
    if overlap:raise AssertionError(f'{len(overlap)} leaking structure identities')
