# Experiment following the 0.179 leaderboard submission

The reported leaderboard result was for the extended COCONUT ensemble, with a roughly 30-minute Kaggle run. Hidden labels are unavailable locally.

## Findings

The original NP-domain validation excluded reference spectra but loaded a fingerprint model fitted on all training structures, including validation identities. `validate_np_holdout.py` now refits that model excluding all 250 NP-domain identities, even when querying a subset. The corrected original-key diagnostic is 0.18038 MRR@25 on 246 eligible structures (`np_holdout_clean.json`). Final validation also scores emitted SMILES by tautomer-canonical identity rather than trusting external raw keys.

All 729,597 `pop` values in the extended database were zero. The ensemble's popularity term therefore supplied no information. COCONUT CSV Lite independently provides annotation levels and multiple records for some connectivity keys. The new fixed prior is:

```
max(annotation_level) + 0.5 * log1p(number of COCONUT records for the connectivity key)
```

Record counts include stereoisomers and must not be described as measured abundance or literature citation counts. Annotation and record counts are joined using the first 14 characters of the published InChIKey; missing/tautomer-mismatched joins receive zero. Candidate chemistry and identities are preserved. No held-out spectrum or spectrum label enters this prior. The existing >=0.999 spectral match gate is retained; otherwise the prior ranks the mass-window candidates, with mass error breaking ties.

## Experiments

Exploratory raw-key results are saved in `experiments/` and reproducible with `scripts/experiments/probe_*.py` (run from project root, `PYTHONPATH=.deps:.`, `OPENBLAS_NUM_THREADS=4`).

- Spectrum analog transfer: 651,166 reference spectra, all NP identities excluded, 0.127–0.137 MRR; rejected.
- Larger polarity-separated ridge: 233,070 structure/adduct observations, 1 or 2 Da bins, alpha 2 or 20, three similarity functions; best aggregate MRR 0.123; rejected.
- COCONUT priors: six variants including mass-only. The combined annotation/count variant had the highest development MRR (0.4900), with 0.5718 on the check partition and 0.5289 overall before final output deduplication/scorer identity conversion.

Development/check membership is `stable_uniform(key, seed=42) < 0.5`. The NP set was already used for earlier model development, and exploratory scripts printed both partitions. This is a development/check diagnostic, not a pristine untouched holdout. Final pipeline metrics and bootstrap intervals are in `np_prior_validation.json`; the same canonical scoring for the former ensemble is in `np_holdout_clean_canonical.json`.

The final emitted-SMILES score for the new prior is **0.55302 MRR@25**, with R@1 **0.43496**, R@25 **0.90244**, development MRR **0.50035**, and check MRR **0.61110**. The query-bootstrap 95% interval is approximately **[0.50068, 0.60829]**, conditional on this development dataset. It does not quantify uncertainty about the hidden-test distribution. Canonical oracle reporting and candidate exclusion now also use tautomer-equivalent identities; raw provider keys previously left some equivalent structures searchable in the nominal outside-pool diagnostic.

Final canonical candidate-oracle recall is 1.0 in the closed-universe experiment. Removing all held identities, including tautomer equivalents, gives exactly 0.0 oracle recall and MRR. All nine unit tests pass. The packaged notebook's real code cells completed offline locally in **32.83 seconds**, including integrity checks; inference and output validation took **31.13 seconds**. The 400-row placeholder submission has no invalid SMILES or duplicate equivalent candidates. These timings are local, not measured Kaggle timings. Artifacts: `submission.np_prior.csv`, `submission.np_prior.stats.json`, and `artifacts/np_prior_notebook_smoke.log`.

## Limits

The 246-query NP set is small and specifically composed of natural products. The prior may perform worse on synthetic compounds or rare, sparsely documented natural products. Held-out structures remain in the searchable candidate database: this measures ranking in that candidate universe, not unknown hidden-test coverage. The visible test set is a training-example placeholder and cannot estimate leaderboard score. A local result above 0.5 does not establish a hidden-test score above 0.5; the new notebook must be rerun/submitted on Kaggle.

## Reproduce the candidate submission

```bash
export PYTHONPATH="$PWD/.deps:$PWD"
export OPENBLAS_NUM_THREADS=4
python scripts/enrich_coconut_prior.py
python scripts/validate_np_holdout.py --config configs/np_prior.json --output notes/np_prior_validation.json
python -m unittest discover -s tests -v
python scripts/package.py --output bundle_np_prior --candidates candidates_np_prior.parquet
python scripts/smoke_notebook.py --bundle bundle_np_prior --output submission.np_prior.csv
```

Upload the contents of `bundle_np_prior/` as the private bundle dataset. Attach that bundle and the competition data to `notebooks/kaggle_submission.ipynb` (exactly one CASMI bundle attached), save/run with internet disabled, and submit the notebook's generated `submission.csv`. The notebook selects `np_prior.json` when `candidates_np_prior.parquet` is present. All 400 locally generated rows are identical to the previous submission because the visible placeholder spectra retain their exact library matches. The changed ranking acts on spectra without those exact matches. Submit a fresh Kaggle notebook run with the new bundle; the unchanged local CSV cannot demonstrate or deliver the hidden-test improvement.
