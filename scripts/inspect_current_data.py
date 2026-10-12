"""Fresh bounded-memory inspection; no RDKit import or historical scores needed."""
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    output = {}
    for name in ('train', 'test'):
        parquet = pq.ParquetFile(f'{name}.parquet')
        nulls = Counter()
        energies = Counter()
        lengths = []
        bad_lengths = nonfinite = 0
        columns = ['ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            for col in columns:
                nulls[col] += batch.column(col).null_count
            data = batch.to_pydict()
            for mz, intensity, energy in zip(*(data[c] for c in columns)):
                if mz is not None:
                    lengths.append(len(mz))
                if mz is None or intensity is None or len(mz) != len(intensity):
                    bad_lengths += 1
                else:
                    nonfinite += int(not np.isfinite(mz).all() or not np.isfinite(intensity).all())
                if energy is not None:
                    energies.update(str(x) for x in energy)
            if len(lengths) % 131072 == 0:
                print(name, len(lengths), 'rows scanned', flush=True)
        output[name] = {'array_nulls': dict(nulls), 'peak_count': pd.Series(lengths).describe(
            percentiles=[.01, .5, .95, .99]).to_dict(), 'bad_array_lengths': bad_lengths,
            'nonfinite_array_rows': nonfinite, 'collision_energy_values': dict(energies)}
        print(name, 'array scan complete', flush=True)
    meta = pd.read_parquet('artifacts/train_metadata.parquet')
    structures = meta[['normalized_smiles', 'molecular_formula', 'inchikey14']].drop_duplicates()
    elements = Counter(); sizes = []; ranges = Counter(); unsupported = []
    for formula in structures.molecular_formula:
        tokens = re.findall(r'([A-Z][a-z]?)(\d*)', str(formula))
        if ''.join(a+b for a,b in tokens) != formula:
            unsupported.append(formula); continue
        counts = Counter()
        for element, n in tokens:
            counts[element] += int(n or 1)
        elements.update(counts.keys())
        for element, n in counts.items():
            ranges[element] = max(ranges[element], n)
        sizes.append(sum(n for e,n in counts.items() if e != 'H'))
    output['structure_formula_inventory'] = {'records': len(structures),
        'element_structure_counts': dict(elements), 'max_element_counts': dict(ranges),
        'heavy_atoms_from_formula': pd.Series(sizes).describe(percentiles=[.01,.5,.95,.99]).to_dict(),
        'unsupported_formula_count': len(unsupported), 'unsupported_examples': unsupported[:20],
        'common_formulas': structures.molecular_formula.value_counts().head(20).to_dict(),
        'supplied_keys_with_multiple_smiles': int((structures.groupby('inchikey14').normalized_smiles.nunique()>1).sum())}
    submission = pd.read_csv('sample_submission.csv')
    test_ids = set(pd.read_parquet('test.parquet', columns=['molecule_id']).molecule_id)
    output['submission'] = {'shape': list(submission.shape), 'columns': list(submission.columns),
        'nulls': submission.isna().sum().to_dict(), 'duplicate_ids': int(submission.molecule_id.duplicated().sum()),
        'ids_match_test': set(submission.molecule_id) == test_ids,
        'guess_counts': submission.smiles.str.split(';').map(len).describe().to_dict()}
    Path('notes/current_data_audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('Wrote notes/current_data_audit.json', flush=True)


if __name__ == '__main__':
    main()
