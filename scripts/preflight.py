"""Check prerequisites before launching expensive preprocessing or training."""
import importlib
import json
import sys
from pathlib import Path


def main():
    checks = {}
    for name in ('numpy', 'pandas', 'pyarrow', 'rdkit.Chem', 'torch'):
        try:
            module = importlib.import_module(name)
            checks[name] = {'ok': True, 'version': getattr(module, '__version__', None)}
            if name == 'torch':
                checks[name]['cuda'] = module.cuda.is_available()
        except Exception as error:
            checks[name] = {'ok': False, 'error': str(error)}
    for name in ('train.parquet', 'test.parquet', 'sample_submission.csv',
                 'artifacts/candidates.parquet', 'artifacts/structure_map.parquet',
                 'artifacts/library_peaks.npy'):
        checks[name] = {'ok': Path(name).exists()}
    Path('notes').mkdir(exist_ok=True)
    Path('notes/current_preflight.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    return 0 if all(v['ok'] for v in checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
