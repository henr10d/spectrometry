import re
from functools import lru_cache
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
ELECTRON=0.000548579909065
PT=Chem.GetPeriodicTable()
TAUT=rdMolStandardize.TautomerEnumerator()
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def formula_mass(formula):
    tokens=re.findall(r'([A-Z][a-z]?)(\d*)',formula)
    if ''.join(a+b for a,b in tokens)!=formula: raise ValueError(f'Invalid formula {formula}')
    return sum(PT.GetMostCommonIsotopeMass(a)*int(b or 1) for a,b in tokens)

@lru_cache(None)
def adduct_parameters(adduct):
    m=re.fullmatch(r'\[(\d*)M((?:[+-][^+-]+)*)\](\d*)([+-])',adduct)
    if not m: raise ValueError(f'Unsupported adduct {adduct}')
    multiplicity=int(m[1] or 1); charge=int(m[3] or 1)*(1 if m[4]=='+' else -1)
    shift=0.
    for sign,term in re.findall(r'([+-])([^+-]+)',m[2]):
        t=re.fullmatch(r'(\d*)([A-Z].*)',term)
        if not t: raise ValueError(adduct)
        shift+=(1 if sign=='+' else -1)*int(t[1] or 1)*formula_mass(t[2])
    return multiplicity,charge,shift-charge*ELECTRON

def neutral_mass(mz,adduct):
    mult,z,shift=adduct_parameters(adduct)
    return (np.asarray(mz)*abs(z)-shift)/mult

@lru_cache(100000)
def identity(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: raise ValueError(f'Invalid SMILES: {smiles}')
    m=TAUT.Canonicalize(m)
    key=Chem.MolToInchiKey(m)
    if not key: raise ValueError('InChI conversion failed')
    return Chem.MolToSmiles(m,isomericSmiles=False),key[:14]

def describe(smiles):
    s,k=identity(smiles); m=Chem.MolFromSmiles(s)
    return {'smiles':s,'key':k,'mass':Descriptors.ExactMolWt(m),'formula':rdMolDescriptors.CalcMolFormula(m),
            'scaffold':MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False),
            'fingerprint':np.packbits(FP.GetFingerprintAsNumPy(m)).tobytes()}

def deduplicate(smiles,limit=25):
    out=[]; seen=set()
    for s in smiles:
        canonical,key=identity(s)
        if key not in seen: out.append(canonical);seen.add(key)
        if len(out)>=limit: break
    return out
