"""Merge train-derived candidates with external structures (COCONUT CSV Lite, etc.)."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from src.chemistry import describe

FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

SMILES_COLUMNS = ("canonical_smiles", "smiles", "SMILES", "standard_smiles")
REQUIRED_COLUMNS = ("smiles", "key", "mass", "formula", "scaffold", "fingerprint")


def worker(s):
    try:
        return {"normalized_smiles": s, **describe(s), "error": None}
    except Exception as e:
        return {"normalized_smiles": s, "error": str(e)}


def coconut_worker(row):
    """Use COCONUT's published identity/descriptors; only compute model inputs."""
    s,key,mass,formula,scaffold=row
    try:
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError('invalid SMILES')
        return {"normalized_smiles":s,"smiles":Chem.MolToSmiles(mol,isomericSmiles=False),
                "key":str(key)[:14],"mass":float(mass),"formula":formula,
                "scaffold":scaffold if isinstance(scaffold,str) else "",
                "fingerprint":np.packbits(FP.GetFingerprintAsNumPy(mol)).tobytes(),"error":None}
    except Exception as e:
        return {"normalized_smiles":s,"error":str(e)}


def read_smiles(path, column=None, chunksize=50000):
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        col = column or next((c for c in SMILES_COLUMNS if c in df.columns), None)
        if not col:
            raise ValueError(f"No smiles column in {path}")
        for s in df[col].dropna().astype(str).unique():
            yield s
        return
    col = column
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if col is None:
            col = next((c for c in SMILES_COLUMNS if c in chunk.columns), None)
            if not col:
                raise ValueError(f"No smiles column in {path}; tried {SMILES_COLUMNS}")
        for s in chunk[col].dropna().astype(str).unique():
            yield s


def normalize_preprocessed(frame, source):
    missing=set(REQUIRED_COLUMNS)-set(frame.columns)
    if missing:
        raise ValueError(f"Preprocessed candidate file is missing {sorted(missing)}")
    frame=frame.dropna(subset=["smiles", "key", "mass"]).copy()
    frame["source"]=frame.get("source", source).fillna(source) if "source" in frame else source
    frame["pop"]=pd.to_numeric(frame.get("pop", 0.0), errors="coerce").fillna(0.0) if "pop" in frame else 0.0
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts/candidates.parquet")
    ap.add_argument("--external", required=True, help="COCONUT CSV/parquet or preprocessed parquet")
    ap.add_argument("--column", default=None)
    ap.add_argument("--output", default="artifacts/candidates_extended.parquet")
    ap.add_argument("--popularity", default=None, help="Optional CSV with key,pop columns (Kaggle pool_popularity)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--source", default="coconut")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N new external SMILES (0=all)")
    ap.add_argument("--use-source-descriptors", action="store_true", help="Fast COCONUT path using its standard InChIKey, exact mass and formula")
    args = ap.parse_args()
    assert rdkit.__version__ == "2026.03.3", rdkit.__version__
    start = time.time()
    base = normalize_preprocessed(pd.read_parquet(args.base), "train")
    if "source" not in base.columns:
        base = base.assign(source="train")
    if "pop" not in base.columns:
        base = base.assign(pop=0.0)
    known = set(base.key)
    known_smiles = set(base.smiles)

    ext_path = Path(args.external)
    parquet_columns=set(pq.ParquetFile(ext_path).schema.names) if ext_path.suffix == ".parquet" else set()
    if "key" in parquet_columns:
        extra = normalize_preprocessed(pd.read_parquet(ext_path), args.source)
        merged = (
            pd.concat([base, extra[~extra.key.isin(known)]], ignore_index=True)
            .sort_values(["mass", "key"])
            .drop_duplicates("key", keep="first")
            .reset_index(drop=True)
        )
    else:
        if args.use_source_descriptors:
            if ext_path.suffix=='.parquet':raw=pd.read_parquet(ext_path)
            else:raw=pd.read_csv(ext_path,low_memory=False)
            required={'canonical_smiles','standard_inchi_key','exact_molecular_weight','molecular_formula','murcko_framework'}
            missing=required-set(raw.columns)
            if missing:raise ValueError(f"Fast COCONUT path is missing {sorted(missing)}")
            raw=raw.dropna(subset=['canonical_smiles','standard_inchi_key','exact_molecular_weight']).drop_duplicates('canonical_smiles')
            raw=raw[~raw.canonical_smiles.isin(known_smiles)]
            if args.limit:raw=raw.iloc[:args.limit]
            todo=list(raw[['canonical_smiles','standard_inchi_key','exact_molecular_weight','molecular_formula','murcko_framework']].itertuples(index=False,name=None))
            work=coconut_worker
        else:
            todo = []
            seen_smiles=set(known_smiles)
            for s in read_smiles(ext_path, args.column):
                if s in seen_smiles:
                    continue
                todo.append(s)
                seen_smiles.add(s)
                if args.limit and len(todo) >= args.limit:
                    break
            work=worker
        print("New external SMILES to canonicalize:", len(todo), flush=True)
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for i, r in enumerate(pool.map(work, todo, chunksize=128)):
                rows.append(r)
                if i and i % 20000 == 0:
                    print(i, len(todo), round(time.time() - start), flush=True)
        m = pd.DataFrame(rows)
        ok = m[m.key.notna()].copy()
        ok["source"] = args.source
        ok["pop"] = 0.0
        merged = (
            pd.concat([base, ok], ignore_index=True)
            .sort_values(["mass", "key"])
            .drop_duplicates("key", keep="first")
            .reset_index(drop=True)
        )

    if args.popularity:
        pop_path=Path(args.popularity)
        pop = pd.read_parquet(pop_path) if pop_path.suffix=='.parquet' else pd.read_csv(pop_path)
        key_col = "key" if "key" in pop.columns else "scorer_inchikey14_rdkit_2026_03_3"
        if key_col not in pop.columns:
            key_col = next(c for c in pop.columns if "inchikey" in c.lower() or c == "key")
        pop_col="pop" if "pop" in pop.columns else next((c for c in pop.columns if "pop" in c.lower() or "count" in c.lower()),None)
        if pop_col is None:raise ValueError("Popularity table has no pop/count column")
        pop = pop[[key_col, pop_col]].rename(columns={pop_col:"pop"}).drop_duplicates(key_col)
        pop_map = dict(zip(pop[key_col], pop["pop"]))
        merged["pop"] = merged["key"].map(pop_map).fillna(merged["pop"]).astype(float)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    index_path=out.with_suffix('.mass_index.npz')
    np.savez_compressed(index_path,mass=merged.mass.to_numpy(dtype=np.float64),row=np.arange(len(merged),dtype=np.int32))
    meta = {
        "base_rows": len(base),
        "output_rows": len(merged),
        "external_added": len(merged) - len(base),
        "seconds": time.time() - start,
        "rdkit": rdkit.__version__,
        "output": str(out),
        "mass_index": str(index_path),
    }
    Path("notes/candidate_merge.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
