"""Create a self-contained local bundle for attachment as a private Kaggle Dataset.
Does not upload, publish, or submit anything.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='bundle')
    ap.add_argument('--candidates',default=None)
    args=ap.parse_args()
    root=Path(args.output);root.mkdir(exist_ok=True)
    for folder in ['src','configs']:
        shutil.copytree(folder,root/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    (root/'artifacts').mkdir(exist_ok=True)
    candidate_name=args.candidates or ('candidates_extended.parquet' if Path('artifacts/candidates_extended.parquet').exists() else 'candidates.parquet')
    for name in [candidate_name,'structure_map.parquet','train_metadata.parquet','library_peaks.npy']:
        shutil.copy2(Path('artifacts')/name,root/'artifacts'/name)
    index=Path('artifacts')/candidate_name.replace('.parquet','.mass_index.npz')
    if index.exists():shutil.copy2(index,root/'artifacts'/index.name)
    if Path('artifacts/fingerprint_ridge.npz').exists():shutil.copy2('artifacts/fingerprint_ridge.npz',root/'artifacts/fingerprint_ridge.npz')
    if Path('wheels').exists():shutil.copytree('wheels',root/'wheels',dirs_exist_ok=True)
    shutil.copy2('requirements.txt',root/'requirements.txt')
    (root/'provenance').mkdir(exist_ok=True)
    for name in ['candidate_build.json','candidate_merge.json','library_build.json','environment.json','np_holdout_proxy.json','coconut_prior.json','np_prior_validation.json','np_holdout_clean_canonical.json','score_improvement.md']:
        source=Path('notes')/name
        if source.exists():shutil.copy2(source,root/'provenance'/name)
    manifest={}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='manifest.json':
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            manifest[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Packaged',sum(x['bytes'] for x in manifest.values())/1e9,'GB')
if __name__=='__main__':main()
