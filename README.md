# CASMI 2026: inspected data and offline baseline

## Current checkout audit (2026-10-12)

See [DATA_ANALYSIS.md](DATA_ANALYSIS.md) for the current execution status. Historical
results below are retained as provenance; this checkout initially lacked the candidate
database, structure map, spectrum cache, and trained weights. The Windows Python
environment is `.venv`. Run commands from the repository root with `$env:PYTHONPATH='.'`.

```powershell
.\.venv\Scripts\python.exe scripts/preflight.py
.\.venv\Scripts\python.exe scripts/analyze_data.py
.\.venv\Scripts\python.exe scripts/inspect_current_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Experimental phase-2 utilities are `src/formulas.py` and `src/fragmentation.py`.
After preflight and tests pass, run `scripts/enumerate_fragments.py --rows 128`
for a bounded pilot with cached graph cuts, ambiguous peak labels, and a coverage
summary. These components have not been promoted to inference or shown to improve
validation. Formula search rejects unsupported elemental/charge assumptions and
fails explicitly when its node budget is exhausted. Fragment matching currently
supports singly charged H-shift hypotheses, not a full ion-rearrangement simulator.

## Nová varianta po skóre 0,179

`configs/np_prior.json` přidává řazení podle veřejných anotací a počtu záznamů COCONUT. Původní `pop` sloupec obsahoval pouze nuly. Nový postup, měření, oprava úniku validačních struktur do fingerprint modelu a omezení jsou v [notes/score_improvement.md](notes/score_improvement.md). Pro tuto variantu použijte **`bundle_np_prior/`** a aktualizovaný `notebooks/kaggle_submission.ipynb`. Na 246 NP strukturách bez jejich referenčních spekter vychází nová varianta na **0,553 MRR@25** oproti **0,180** původního ensemble po opravě trénovacího úniku. Lokální validace není záruka skóre na skrytém testu.

```bash
PYTHONPATH=.deps:. python scripts/enrich_coconut_prior.py
PYTHONPATH=.deps:. OPENBLAS_NUM_THREADS=4 python scripts/validate_np_holdout.py --config configs/np_prior.json --output notes/np_prior_validation.json
python scripts/package.py --output bundle_np_prior --candidates candidates_np_prior.parquet
PYTHONPATH=.deps:. OPENBLAS_NUM_THREADS=4 python scripts/smoke_notebook.py --bundle bundle_np_prior --output submission.np_prior.csv
```

Následující části dokumentují původní baseline.

Start with [notes/data_analysis.md](notes/data_analysis.md). **The visible test set is a documented training-example placeholder, replaced during Kaggle scoring. Local test matches are not a leaderboard estimate.** This is an initial reproducible retrieval/validation milestone, not a claim of a finished maximum-performance system.

## Reproduce locally

Python 3.11–3.13. Install `requirements.txt` in an isolated environment. Local development dependencies are in `.deps`; commands below use them. Inputs remain at the project root.

```bash
export PYTHONPATH="$PWD/.deps:$PWD"
export OPENBLAS_NUM_THREADS=4
python scripts/analyze_data.py
python scripts/build_candidates.py --workers 8
python scripts/build_library.py
python scripts/audit_data.py
python -m unittest discover -s tests -v
python scripts/validate.py --n 128
python scripts/train.py --n-train 40000
python scripts/validate_gate.py
python scripts/validate_novelty.py
python scripts/train.py --n-train 40000 --final
python scripts/coverage.py
python scripts/stratify.py
python scripts/make_submission.py
```

## Rozšířený candidate pool (COCONUT + popularity)

Rozšíření je záměrně oddělené od reprodukovatelného train-only baseline. Stažení vyžaduje internet; následná inference už ne.

```bash
python scripts/download_coconut.py --out-dir external
python scripts/merge_external_candidates.py \
  --external external/coconut_csv_lite-10-2026.csv \
  --popularity external/pool_popularity.csv \
  --output artifacts/candidates_extended.parquet \
  --workers 20 \
  --use-source-descriptors
python scripts/validate_np_holdout.py --config configs/ensemble.json
python scripts/package.py
```

Rychlá COCONUT cesta přebírá jeho publikovaný standard InChIKey, exact mass a formuli, lokálně dopočítá kanonický SMILES a Morgan fingerprint, deduplikuje connectivity, zachová původ (`source`) a vytvoří seřazený mass index. Finální top 25 se znovu deduplikuje soutěžní tautomerickou identitou. Bez `--use-source-descriptors` proběhne pomalejší plná lokální tautomerická kanonikalizace. `validate_np_holdout.py` hlásí zvlášť diagnostický únikový režim, režim bez spekter stejné struktury a simulaci struktury mimo candidate pool; pro každý uvádí oracle recall i conditional MRR. Notebook automaticky použije ensemble jen tehdy, když bundle skutečně obsahuje rozšířený pool.

The full train spectrum cache uses 64 strongest peaks per spectrum, sums identical m/z peaks, and is memory mapped. All spectra remain represented. Validation groups use tautomer-canonical first-block InChIKeys; scaffold splits also exclude all held scaffolds. The candidate table keeps molecular components/charge rather than silently applying salt removal to the metric.

`experiments/results.csv` contains measured MRR@25 and recalls. Closed-universe experiments include validation **structures** as searchable candidates, but exclude their **spectra** globally. They diagnose ranking and mass retention; they do not estimate real external-database coverage. Train-only strict held-out coverage is zero by construction. No random spectrum-split scores are advertised.

## Offline Kaggle notebook

```bash
python scripts/package.py
```

Attach `bundle/` as a **private** Kaggle Dataset and attach competition data. Open `notebooks/kaggle_submission.ipynb`, disable internet, and run. It discovers the attached bundle and competition files, verifies SHA-256 hashes, installs the pinned RDKit wheel offline if required, and writes `/kaggle/working/submission.csv`. Python 3.11 and 3.12 RDKit wheels are included; the standard Kaggle environment must supply NumPy, pandas, SciPy and PyArrow. No training or internet calls occur during inference. Do not publish the bundled competition-derived artifacts without checking competition redistribution rules.

`submission.stats.json` and `submission.audit.csv` report checks, runtime and mass-window fallbacks. Fallbacks guarantee a valid file, not a chemically credible identification. The final Kaggle hardware/runtime is unverified until the notebook is run there. Local notebook execution is recorded separately.

## Limits and next work

No test labels are available. Candidate coverage on test is unknown. External natural-product databases, pretrained encoders, formula inference, forward-spectrum models, calibrated rankers and de novo generation are researched but not implemented in this milestone. A small CPU ridge fingerprint baseline provides an initial learned comparator. See the report for prioritized next steps and compute estimates.

## Selected baseline and learned comparator

The default `configs/baseline.json` retains spectral matches with aggregate cosine >=0.999 and otherwise orders the mass-window candidates by neutral-mass error. This gate improved the strict unseen-structure diagnostics over unconditional cosine. It is a conservative baseline, not a calibrated confidence model for independent library-known spectra. `configs/cosine.json` preserves the original retriever. `configs/fingerprint.json` runs the optional saved ridge model with the same near-exact-match gate; it improved random/scaffold MRR but degraded the natural-product domain holdout, so it was **not** promoted to default.

The natural-product example holdout is used for model comparison, so it is no longer an untouched final assessment. All initial folds are development diagnostics (128 queries each), not leaderboard estimates. `validate_novelty.py` adds 64 queries with every training neighbor above 0.4 Morgan Tanimoto removed. The packaged candidate database is train-derived; adding an independently frozen external database remains the highest-priority next phase.
# spectrometry
