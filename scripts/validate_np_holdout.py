"""Leaderboard-oriented proxy: enveda-np-examples holdout (250 test-instrument structures)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import Retriever
from src.metric import evaluate_keys
from src.chemistry import identity
from src.validation import stable_uniform


def build_queries(r, keys):
    eligible = r.meta[
        r.meta.key.isin(keys)
        & (r.meta.instrument_type == "timsTOF")
        & r.meta.precursor_mz.between(100, 1000)
        & r.meta.ingest_lib.eq("enveda-np-examples")
    ]
    queries = []
    for key in keys:
        rows = eligible[eligible.key == key].index.to_numpy()
        rows = rows[np.argsort([str(r.adduct[x]) for x in rows], kind="stable")][:9]
        q = r.meta.iloc[rows].copy()
        q["ms2_mzs"] = [r.peaks[x, 0] for x in rows]
        q["ms2_normalized_intensities"] = [r.peaks[x, 1] for x in rows]
        queries.append(q)
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ensemble.json")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n", type=int, default=0, help="Subsample queries (0 = all NP examples in train)")
    ap.add_argument("--output", default="notes/np_holdout_proxy.json")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--protocol", choices=['all', 'all_visible', 'no_own_spectra', 'outside_candidate_pool'], default='all')
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    r = Retriever(args.artifacts, cfg)
    eligible_mask=(r.meta.ingest_lib.eq("enveda-np-examples")
                   & r.meta.instrument_type.eq("timsTOF")
                   & r.meta.precursor_mz.between(100,1000)
                   & r.meta.key.notna())
    keys=sorted(r.meta.loc[eligible_mask,"key"].unique())
    if args.n:
        keys = sorted(np.random.default_rng(cfg.get("seed", 2026)).choice(keys, min(args.n, len(keys)), replace=False))
    queries = build_queries(r, keys)
    method = cfg.get("method", "ensemble")
    start = time.time()
    # Exclude the complete domain, even when evaluating only a query subsample.
    # Excluding reference spectra alone does not remove learned-model leakage.
    held = set(r.meta.loc[r.meta.ingest_lib.eq("enveda-np-examples"), "key"].dropna())
    training = None
    if method in {"ensemble", "fingerprint", "hybrid"}:
        from scripts.train import fit_model
        model, _, seconds, count = fit_model(r, held, args.n_train, cfg.get("seed", 2026))
        r.fingerprint_model = model
        training = {"excluded_structures": len(held), "training_structures": count,
                    "seconds": seconds, "refit_without_np_domain": True}
    protocols = {
        "all_visible": {},
        "no_own_spectra": {"excluded_keys": held},
        "outside_candidate_pool": {"excluded_keys": held, "candidate_excluded_keys": held},
    }
    if args.protocol != 'all':
        protocols = {args.protocol: protocols[args.protocol]}
    results={}
    for name,kwargs in protocols.items():
        preds=[];oracle=[];counts=[]
        for key,q in zip(keys,queries):
            ids,info=r.predict(q,method=method,canonical_oracle=True,**kwargs)
            # Score emitted SMILES with the same tautomer identity used for
            # output deduplication, not external providers' raw InChIKeys.
            preds.append([identity(r.db.smiles[cid])[1] for cid in ids]);oracle.append(key in info["oracle_keys"]);counts.append(info["candidates"])
        score=evaluate_keys(keys,preds)
        score.update({
            "candidate_oracle_recall":float(np.mean(oracle)),
            "conditional_mrr":float(score["MRR25"]/np.mean(oracle)) if np.any(oracle) else 0.0,
            "median_candidates":float(np.median(counts)),
        })
        ranks = [p.index(k) + 1 if k in p else 0 for k, p in zip(keys, preds)]
        rr = np.array([1 / rank if rank else 0 for rank in ranks])
        dev = np.array([stable_uniform(key, 42) < .5 for key in keys])
        score['selection_split_seed'] = 42
        score['development_MRR25'] = float(rr[dev].mean()) if dev.any() else None
        score['check_MRR25'] = float(rr[~dev].mean()) if (~dev).any() else None
        rng = np.random.default_rng(2026)
        means = rr[rng.integers(0, len(rr), size=(2000, len(rr)))].mean(axis=1)
        score['MRR25_bootstrap_95pct'] = np.quantile(means, [.025, .975]).tolist()
        score['query_ranks'] = dict(zip(keys, ranks))
        results[name]=score
    out = {
        "protocols":results,
        "method": method,
        "n_queries": len(keys),
        "seconds": time.time() - start,
        "config": args.config,
        "candidates_file": cfg.get("candidates_file", "candidates.parquet"),
        "training": training,
        "limitation": "Closed candidate universe: held-out structures remain searchable; hidden-test coverage is unknown.",
    }
    print(json.dumps(out, indent=2))
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
