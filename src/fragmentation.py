"""Connected graph-cut inventories, not a claim of chemically simulated ions.

Hydrogens remain those attached in the parent: sanitizing severed graphs would
silently cap dangling valences and change masses. Explicit H shifts model bounded
rearrangement hypotheses. Cuts are exhaustive only within declared bond/budget limits.
"""
from itertools import combinations
from math import comb
from rdkit import Chem
from .chemistry import ELECTRON, formula_mass
from .formulas import composition, format_formula, ppm_error


def enumerate_fragments(smiles, max_cuts=2, max_bonds=60, max_cut_sets=2000):
    if max_cuts not in (1, 2):
        raise ValueError('Only one/two-cut enumeration is supported')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError('Isotope-labeled fragments require a separate isotope inventory')
    if mol.GetNumBonds() > max_bonds:
        raise ValueError('Molecule exceeds explicit bond budget')
    bonds = [(b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.SINGLE or b.IsInRing()]
    number = sum(comb(len(bonds), k) for k in range(1, max_cuts+1))
    if number > max_cut_sets:
        raise ValueError('Molecule exceeds explicit cut-set budget')
    unique = {}
    for size in range(1, max_cuts+1):
        for selected in combinations(bonds, size):
            removed = {b[0] for b in selected}
            adjacency = [set() for _ in mol.GetAtoms()]
            for b in mol.GetBonds():
                if b.GetIdx() not in removed:
                    a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    adjacency[a].add(c); adjacency[c].add(a)
            unseen = set(range(mol.GetNumAtoms()))
            while unseen:
                stack = [min(unseen)]; group = set()
                while stack:
                    a = stack.pop()
                    if a in group:
                        continue
                    group.add(a); stack.extend(adjacency[a]-group)
                unseen -= group
                if len(group) == mol.GetNumAtoms():
                    continue  # ring opening alone gives no detached fragment
                atoms = tuple(sorted(group))
                boundary = tuple(sorted(b[0] for b in selected if (b[1] in group) != (b[2] in group)))
                key = (atoms, boundary)
                if key in unique:
                    continue
                counts = {}
                charge = 0
                for idx in atoms:
                    atom = mol.GetAtomWithIdx(idx)
                    element = atom.GetSymbol()
                    counts[element] = counts.get(element, 0)+1
                    counts['H'] = counts.get('H', 0)+atom.GetTotalNumHs()
                    charge += atom.GetFormalCharge()
                formula = format_formula(counts)
                unique[key] = dict(atoms=atoms, cuts=tuple(sorted(removed)), boundary=boundary,
                                   formula=formula, mass=formula_mass(formula), charge=charge,
                                   graph=Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms),
                                       bondsToUse=[b.GetIdx() for b in mol.GetBonds() if b.GetIdx() not in removed
                                                   and b.GetBeginAtomIdx() in group and b.GetEndAtomIdx() in group]))
    return list(unique.values())


def match_fragments(mzs, fragments, ppm=10., polarity=1, hydrogen_shifts=(-1, 0, 1)):
    if polarity not in (-1, 1):
        raise ValueError('Only singly charged fragment hypotheses are supported')
    matches = [[] for _ in mzs]
    hmass = formula_mass('H')
    for fi, fragment in enumerate(fragments):
        if fragment['charge'] not in (0, polarity):
            continue
        for shift in hydrogen_shifts:
            if composition(fragment['formula']).get('H', 0) + shift < 0:
                continue
            theoretical = fragment['mass'] + shift*hmass - polarity*ELECTRON
            for pi, observed in enumerate(mzs):
                if observed > 0 and ppm_error(float(observed), theoretical) <= ppm:
                    matches[pi].append(dict(fragment=fi, hydrogen_shift=shift,
                                            ppm=ppm_error(float(observed), theoretical)))
    return matches
