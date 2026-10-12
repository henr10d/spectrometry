"""Run the notebook's real code cells locally with path overrides, internet unnecessary."""
import argparse,json,os
from pathlib import Path
root=Path.cwd()
ap=argparse.ArgumentParser()
ap.add_argument('--bundle',default='bundle')
ap.add_argument('--output',default='submission.csv')
args=ap.parse_args()
os.environ.update(CASMI_BUNDLE=str((root/args.bundle).resolve()),CASMI_DATA=str(root),CASMI_OUTPUT=str((root/args.output).resolve()))
ns={}
for cell in json.loads(Path('notebooks/kaggle_submission.ipynb').read_text())['cells']:
    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'kaggle_submission.ipynb','exec'),ns)
