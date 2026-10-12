"""Add label-independent COCONUT annotation and connectivity-record counts.

Counts represent database records (including stereoisomers), not experimental
abundance. The source-key join is conservative: unmatched tautomers get zero.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def enrich(candidates, raw):
    raw = raw.dropna(subset=['standard_inchi_key']).copy()
    raw['key'] = raw.standard_inchi_key.str[:14]
    raw['annotation_level'] = pd.to_numeric(raw.annotation_level, errors='coerce').fillna(0)
    prior = raw.groupby('key').agg(
        coconut_annotation=('annotation_level', 'max'),
        coconut_records=('key', 'size'),
    )
    result = candidates.drop(columns=list(prior.columns), errors='ignore').copy()
    # Do not replace the candidate identity, chemistry, or original provenance.
    for column in prior:
        result[column] = result.key.map(prior[column]).fillna(0)
    result['coconut_records'] = result.coconut_records.astype(np.int32)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='artifacts/candidates_extended.parquet')
    ap.add_argument('--external', default='external/coconut_csv_lite-10-2026.csv')
    ap.add_argument('--output', default='artifacts/candidates_np_prior.parquet')
    args = ap.parse_args()
    raw = pd.read_csv(args.external, usecols=['standard_inchi_key', 'annotation_level'])
    result = enrich(pd.read_parquet(args.base), raw)
    result.to_parquet(args.output, index=False)
    digest = hashlib.sha256()
    with open(args.external, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    report = {
        'source': args.external, 'source_sha256': digest.hexdigest(),
        'output': args.output, 'candidates': len(result),
        'matched_candidates': int((result.coconut_records > 0).sum()),
        'score': 'max(annotation_level) + 0.5 * log1p(connectivity record count)',
        'uses_spectral_labels': False,
        'limitations': 'Counts include stereoisomers; unmatched tautomer keys receive zero. This is a natural-product prior, not a learned spectral model.',
    }
    Path('notes/coconut_prior.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
