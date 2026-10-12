"""Bounded neutral-formula search. Bounds must be fitted on training only."""
import math
import re
from functools import lru_cache
from .chemistry import PT, formula_mass


def composition(formula):
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    if not tokens or ''.join(a+b for a, b in tokens) != formula:
        raise ValueError(f'Unsupported neutral formula: {formula}')
    result = {}
    for element, count in tokens:
        count = int(count or 1)
        if count < 1:
            raise ValueError(formula)
        result[element] = result.get(element, 0) + count
    return result


def dbe(counts):
    if set(counts) - {'C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'}:
        raise ValueError('DBE model only supports CHNOS and halogens')
    return 1 + counts.get('C', 0) + (counts.get('N', 0) - counts.get('H', 0)
        - sum(counts.get(x, 0) for x in ('F', 'Cl', 'Br', 'I'))) / 2


def ppm_error(observed, candidate):
    if not math.isfinite(observed) or observed <= 0:
        raise ValueError('Observed mass must be finite and positive')
    return 1e6 * abs(observed - candidate) / observed


def format_formula(counts):
    order = ['C', 'H'] + sorted(set(counts) - {'C', 'H'})
    return ''.join(e + (str(counts[e]) if counts[e] != 1 else '')
                   for e in order if counts.get(e, 0))


@lru_cache(maxsize=4096)
def _enumerate(mass, bounds, ppm, max_nodes):
    limits = dict(bounds)
    if 'H' not in limits:
        raise ValueError('Hydrogen bound is required')
    dbe({e: 0 for e in limits})  # reject unsupported valence assumptions
    elements = sorted(set(limits) - {'H'}, key=lambda e: -PT.GetMostCommonIsotopeMass(e))
    tolerance = mass * ppm * 1e-6
    hmass = PT.GetMostCommonIsotopeMass('H')
    nodes = 0
    result = []
    counts = {}

    def visit(i, used):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError('Formula node budget exceeded; increase budget or tighten training bounds')
        if i == len(elements):
            low = max(0, math.ceil((mass-tolerance-used)/hmass))
            high = min(limits['H'], math.floor((mass+tolerance-used)/hmass))
            for h in range(low, high+1):
                counts['H'] = h
                value = dbe(counts)
                # Closed-shell neutral CHNOS/halogen model; radicals/charged formulas need separate handling.
                if value >= 0 and value.is_integer():
                    formula = format_formula(counts)
                    error = ppm_error(mass, formula_mass(formula))
                    if error <= ppm:
                        result.append((formula, error, value))
            return
        e = elements[i]
        emass = PT.GetMostCommonIsotopeMass(e)
        for n in range(min(limits[e], int((mass+tolerance-used)/emass))+1):
            counts[e] = n
            visit(i+1, used+n*emass)
        counts.pop(e, None)

    visit(0, 0.)
    return tuple(sorted(result, key=lambda x: (x[1], x[0]))), nodes


def enumerate_formulas(mass, bounds, ppm=10., max_nodes=2_000_000):
    ppm_error(mass, mass)
    if not math.isfinite(ppm) or ppm < 0 or max_nodes < 1:
        raise ValueError('Invalid tolerance or search budget')
    if any(not isinstance(n, int) or n < 0 for n in bounds.values()):
        raise ValueError('Bounds must be nonnegative integers')
    return _enumerate(float(mass), tuple(sorted(bounds.items())), float(ppm), max_nodes)
