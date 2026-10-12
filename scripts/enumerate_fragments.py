"""Cache bounded graph cuts and ambiguous peak pseudo-labels for a pilot."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from src.fragmentation import enumerate_fragments, match_fragments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='train.parquet')
    parser.add_argument('--config', default='configs/chemistry_pilot.json')
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--output', default='artifacts/fragment_pilot')
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError('--rows must be positive')
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    stats = {'rows': 0, 'skipped': 0, 'peaks': 0, 'one_cut_peaks': 0, 'two_cut_peaks': 0,
             'total_intensity': 0., 'matched_intensity': 0., 'config': config}
    start = time.perf_counter()
    columns = ['normalized_smiles','ms2_mzs','ms2_normalized_intensities','ionization_mode','adduct']
    with (output/'labels.jsonl').open('w', encoding='utf-8') as handle:
        for batch in pq.ParquetFile(args.input).iter_batches(batch_size=min(args.rows, 256), columns=columns):
            for row in batch.to_pylist():
                if stats['rows'] >= args.rows:
                    break
                index = stats['rows']; stats['rows'] += 1
                # Adduct-independent H shift hypotheses are deliberately restricted.
                if row['adduct'] not in ('[M+H]+', '[M-H]-'):
                    stats['skipped'] += 1
                    handle.write(json.dumps({'row':index,'error':'unsupported precursor adduct'})+'\n')
                    continue
                try:
                    token = hashlib.sha256(json.dumps([row['normalized_smiles'],config],sort_keys=True).encode()).hexdigest()
                    cache = output/f'{token}.json'
                    if cache.exists():
                        fragments = json.loads(cache.read_text())
                    else:
                        fragments = enumerate_fragments(row['normalized_smiles'], config['max_cuts'],
                            config['max_bonds'], config['max_cut_sets'])
                        cache.write_text(json.dumps(fragments), encoding='utf-8')
                    mzs = row['ms2_mzs']; intensities = np.asarray(row['ms2_normalized_intensities'])
                    if len(mzs) != len(intensities) or not np.isfinite(intensities).all():
                        raise ValueError('Invalid spectrum arrays')
                    matches = match_fragments(mzs, fragments, config['ppm'],
                        1 if row['ionization_mode']=='positive' else -1, config['hydrogen_shifts'])
                    one = [any(len(fragments[x['fragment']]['cuts'])==1 for x in labels) for labels in matches]
                    two = np.asarray([bool(labels) for labels in matches])
                    stats['peaks'] += len(mzs); stats['one_cut_peaks'] += sum(one)
                    stats['two_cut_peaks'] += int(two.sum())
                    stats['total_intensity'] += float(intensities.sum())
                    stats['matched_intensity'] += float(intensities[two].sum())
                    handle.write(json.dumps({'row':index,'cache':cache.name,'peak_matches':matches})+'\n')
                except ValueError as error:
                    stats['skipped'] += 1
                    handle.write(json.dumps({'row':index,'error':str(error)})+'\n')
            if stats['rows'] >= args.rows:
                break
    stats['seconds'] = time.perf_counter()-start
    stats['limitations'] = 'Ordered pilot sample, not a performance estimate; H shifts are hypotheses, no neutral-loss/adduct cascade model.'
    (output/'summary.json').write_text(json.dumps(stats,indent=2), encoding='utf-8')
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
