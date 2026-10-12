# CASMI 2026 inspection and implementation plan

Inspected 2026-09-23. This report distinguishes measurements, proposals and unresolved checks. Machine-readable full schema/counts: `data_statistics.json`. No original data files are modified.

## Critical finding: visible test is a placeholder

The [official data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) explicitly says the visible test file consists of training examples and will be replaced by the hidden test during rerun (approximately the same size). Original float64 peak masses and intensities of initial inspected test spectra matched enveda-180 training rows. Near-perfect local test spectral matches are therefore expected and **not a leaderboard estimate**. Do not infer hidden-test mass limits from the 400 visible examples, hard-code predictions, or package a test-specific candidate subset. The baseline retains all training spectra/candidates and handles test IDs dynamically. Validation uses a broader 100–1000 m/z window and includes an NP domain holdout. Hidden class proportions and exact structure coverage remain unknown.

## Files and schema

The directory initially contained only train.parquet (3,033,286,496 bytes), test.parquet (4,848,729 bytes), sample_submission.csv (43,619 bytes), and a 3,032,290,079-byte ZIP containing those same three filenames. No existing code, models, candidate databases or repository instructions were found.

Training: **2,539,608 spectra, 275,810 supplied InChIKey14 values, 277,566 SMILES, 51,291 formulas**. Test: **1,213 spectra, 400 molecules**. Sample submission: 400 rows, columns `molecule_id,smiles`, 25 repeated ethanol placeholders per row. These placeholders are not candidates or labels.

Train columns: ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula, ionization_mode, instrument_type, adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs, ms2_normalized_intensities, num_peaks, base_peak_intensity, collision_energy_ev, collision_energy_orig, collision_energy_orig_units. Test columns: molecule_id, spectrum_id, the two peak lists, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz and the three collision energy fields. **No test formula, MS1 isotope envelope, retention time, SMILES or source library is provided.** Train has no molecule_id: group by canonical structure, not rows.

Peak masses/intensities and converted energies are variable-length double lists. Intensities appear base-peak normalized; validate all arrays before modeling. Do not treat precursor-minus-fragment m/z as a true neutral loss across arbitrary charges. Test adducts are singly charged, simplifying this branch.

## Distributions and chemistry

Test spectra/molecule: mean 3.0325, median 3, range 1–9. Train spectra/supplied connectivity: mean 9.208, median 6, max 1,468. Test precursor m/z 245.0931–460.1554, median 329.1978; train median 350.1088, 99th percentile 976.3292, maximum **2,430,333**. Train precursor errors have a severe outlier tail (99th percentile 2,884 ppm); do not let it dictate test mass tolerances. Use exact structure masses, audit errors by source/adduct, and retain quality flags.

All test spectra are **timsTOF, eV**. Positive/negative counts: 987/226. Test adducts: [M+H]+ 959; [M-H]- 193; [M+CH2O2-H]- 31; [M+Na]+ 22; [M+NH4]+ 4; [M+K]+ 2; [M+Cl]- 2. Collision energies include individual 20/40/60 and merged 20,40,60; negative mode original strings have minus signs. Converted list semantics must be checked rather than treating sign as negative physical energy.

Train is 1,899,640 positive / 639,968 negative. Large dimer populations include [2M+Na]+ 190,033; [2M+H]+ 92,041; [2M-H]- 56,172. Implement a compositional adduct parser with molecule multiplicity, signed charge, elemental mass shifts and electron correction; unsupported annotations must fail explicitly.

| Training source | Spectra | Supplied structures |
|---|---:|---:|
| enveda-180 | 1,153,785 | 182,941 |
| pluskal_ms2 | 527,581 | 46,821 |
| riken | 347,171 | 15,892 |
| gnps | 220,849 | 45,750 |
| massbank | 101,727 | 9,180 |
| mona | 92,416 | 11,681 |
| spectraverse | 50,933 | 9,631 |
| msdial | 40,765 | 9,127 |
| drug_plus | 2,545 | 2,539 |
| enveda-np-examples | 1,184 | 250 |
| masaryk | 652 | 416 |

The 250 natural-product examples are particularly useful for domain validation, but too few for repeated unrestricted tuning. Train includes 1,154,969 timsTOF spectra. Instrument names otherwise require normalization (QTOF/qtof/qTof, etc.). Do not use library IDs as chemistry predictors.

## Leakage and validation

Random spectrum splits are invalid. Canonicalize tautomers with RDKit and use first-block InChIKey groups across every source; supplied keys are useful provenance but not automatically the scorer's tautomer-normalized key. Do not strip salts or neutralize in the metric without evidence the official scorer does so. Candidate standardization is a separate policy.

Implement random structure and Murcko-scaffold group splits with all spectra of held-out identities excluded from reference evidence and model fitting. Keep ringless molecules together for strict scaffold validation. Add enveda-np-examples domain holdout with its identities removed globally. A later fingerprint-distance holdout is needed because different scaffolds can still be close analogs.

Report two candidate protocols: (1) train-only candidates, which necessarily have zero exact coverage on strictly unseen structures, and (2) explicitly labeled closed-universe structure-only candidates, containing held-out structures but no held-out spectral evidence. Protocol (2) measures ranking conditional on coverage and is **not** an external-database coverage estimate. Ultimately freeze an independently sourced database and report its actual coverage, plus results stratified by maximum training fingerprint similarity. Hidden test labels are unavailable: mass matches alone are not evidence of exact hidden-test coverage. The visible-test overlap is an explicitly documented placeholder, as discussed above.

Hash exact peak arrays and metadata to audit train/test and cross-library duplicates. IDs look opaque; no established ID shortcut. Merged-energy and individual-energy observations may share peaks, so multiple spectra are correlated evidence. Compare mean versus max scoring on validation rather than counting all measurements as independent.

## Three solution levels and planning estimates

These are estimates, not measured GPU benchmarks. Local machine: ~67 GB RAM, ~210 GB disk free at inspection, PyTorch CUDA unavailable. RDKit was absent and installed into a temporary isolated dependency directory for development.

| Level | Approach | Offline preparation estimate | Offline inference estimate | Storage/RAM estimate |
|---|---|---|---|---|
| Fast | Adduct-aware mass lookup; direct binned cosine and neutral-loss cosine over measured library; aggregate per molecule | 0.5–3 CPU hours | 5–60 CPU minutes for present 400 molecules | 1–5 GB artifacts, 4–12 GB RAM |
| Strong (recommended) | Expanded NP/general candidate DB; fingerprint predictor using binned peaks/losses then small peak transformer; mass-hard negatives; validated spectral evidence fusion | 1–3 GPU days plus 2–12 CPU hours database work | 0.5–3 GPU hours, aim below 6 | 5–30 GB artifacts; 12–24 GB RAM, 8–16 GB VRAM |
| Maximum | DreaMS/MIST-derived ensemble, multi-formula hypotheses, cached ICEBERG forward spectra for top candidates, calibrated reranker, limited de novo generation | 3–14 GPU days, larger precompute | 2–6 GPU hours only after profiling/capping | 20–100 GB artifacts, 16–32 GB RAM; 16 GB+ VRAM |

For scale, 1M packed 2048-bit fingerprints need 256 MB; 1M float16 1024-D embeddings need 2.048 GB before metadata. FAISS is unnecessary for a few hundred candidates per mass bin. Precompute embeddings outside submission. Do not rerun training in submission.

## External resources: expected value and deployability

Research sources checked on 2026-09-23. No external artifact is silently assumed approved. The competition overview/data pages confirm molecule-level ranked output, but the rules page returned no readable body; a subsequent targeted search of the official overview confirmed CPU/GPU <=9 hours, internet disabled and freely/publicly available external data including pretrained models. Full supplemental rules still need review. No redistribution of competition data is performed.

1. **COCONUT / LOTUS structure snapshots**: highest first value because coverage limits rankers. COCONUT September 2026 has a 191 MB CSV-lite ZIP and states CC0; LOTUS states CC0. Preserve snapshot/hash, source IDs, licenses; evaluate incremental held-out coverage before training. [COCONUT downloads](https://coconut.naturalproducts.net/download), [LOTUS](https://lotus.nprod.net/).
2. **PubChem**: broaden only after NP coverage measurement; bulk structures can be huge. Mass-indexed local subset is feasible, whole raw dump may exceed this machine's free disk. Contributor terms can vary; do not assume all annotations have one blanket license. [Official download/licensing guidance](https://pubchem.ncbi.nlm.nih.gov/docs/downloads).
3. **DreaMS (2025)**: MIT code, pretrained model links, 1024-D spectrum embeddings; attractive frozen-encoder experiment. Check checkpoint license and pretraining overlap separately. It does not itself supply a complete molecule decoder. [Code](https://github.com/pluskal-lab/DreaMS), [weights](https://zenodo.org/records/10997887).
4. **MIST / MIST-CF**: MIT MIST code, fingerprint/contrastive retrieval and precursor-formula tools. Formula inference matters here because test formulas are absent. Legacy CUDA environments require porting and offline wheels. MIST explicitly cannot redistribute CSI2022, so do not bundle NIST-derived training data. [MIST](https://github.com/samgoldman97/mist).
5. **MassSpecGym (2024)**: useful evaluation infrastructure and public baselines; avoid oracle-formula settings as competition estimates. Deduplicate against existing sources before expecting more training diversity. [Repository](https://github.com/pluskal-lab/MassSpecGym).
6. **GNPS / MassBank / MoNA / historical CASMI**: much is already represented; incremental benefit requires overlap audit. Direct GNPS contributions default CC0, imported collections may differ. Check per-record MassBank licenses and MoNA source terms. [GNPS policy](https://ccms-ucsd.github.io/GNPSDocumentation/browselibraries/), [MassBank records including CASMI](https://github.com/MassBank/MassBank-data).
7. **ICEBERG / ms-pred**: candidate forward-spectrum reranking is plausible with caching; first benchmark top 50–100 on actual timsTOF data and adducts. [Code](https://github.com/coleygroup/ms-pred). No model integrated yet.
8. **FRIGID (2026)**: released diffusion generation code with fingerprint/formula conditioning and ICEBERG refinement. Published repository results use known formula; that assumption does not hold here. Check full license/weight availability, formula errors and Kaggle hardware timing before considering slots. [Repository](https://github.com/coleygroup/FRIGID). SpecBridge (2026) is another research lead, but deployment was not verified.
9. HMDB, NIST, SIRIUS services: defer until explicit competition-compatible redistribution/offline rights are verified. Uni-Mol/ChemBERTa add complexity without solving spectral domain shift or coverage. Spec2Vec/MS2DeepScore are reasonable embedding comparisons after the cheaper baseline. Do not integrate every named model without measured incremental MRR.

## Implementation order

1. Finish schema/data quality audit; lock chemistry identity, metric, submission assertions and seeded grouped splits.
2. Build canonical train candidate table; measure mass-filter retention and honest train-only coverage. Run mass, binned cosine and neutral-loss baselines.
3. Execute a full offline submission and measure actual time/memory. Preserve the working baseline.
4. Freeze external NP candidate snapshot and measure oracle coverage. Expand with LOTUS/PubChem only where useful.
5. Train a compact fingerprint predictor, then compare pretrained embeddings; use mass/formula hard negatives and molecule-balanced training. Train-derived features only.
6. Tune ensemble/reranker on development folds, confirm on untouched holdout. Evaluate formula inference with predicted formulas, never validation truth as test input.
7. Add forward prediction/generation only when measured marginal recall/MRR justifies runtime. Package source, pinned wheels, weights, indices, hashes, licenses; test Kaggle notebook with internet off.

The first milestone is a valid, measured retrieval baseline, not a claim of competitive unseen-molecule identification. Subsequent results and limitations will be appended below.

## Metric version confirmation

The [official overview](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) specifies RDKit **2026.03.3**. Development initially downloaded 2026.03.6; final candidate preprocessing and inference must use 2026.03.3. No claim of exact reproduction of unpublished scorer code is made beyond the published canonicalization procedure.

## Completed full-array audit

`quality_audit.json` records a 385.8-second full scan. All **1,213** visible test peak/intensity arrays have byte-identical float64 matches in enveda-180, associated with **400** supplied structure keys. This is the documented placeholder behavior. Zero train length mismatches, empty spectra or nonfinite peak values were found. There are 682 train rows with duplicate exact m/z peaks, merged by the baseline. Visible test has no repeated exact m/z rows or duplicate spectra. It has 13,572 peaks above precursor+0.02 Da; these may include isotopes/noise and are not treated as negative neutral losses. Test peak counts: median 230, max 3,259. Converted negative-mode collision energies are nonnegative. Across visible spectra of each molecule, inferred neutral-mass spread is median 0 ppm, 95th percentile 2.70 ppm, maximum 5.50 ppm. This supports a 10 ppm initial diagnostic window, not proof of hidden-test mass accuracy.

The top-64-peak cache covers all 2,539,608 training rows, built in 404.8 seconds on this machine. Its float32 storage is approximately 1.30 GB.

## Updated forward-model availability

The current [ms-pred repository](https://github.com/coleygroup/ms-pred) lists ICEBERG 2.1 and GLACIER (July 2026). MIT code does not imply unrestricted NIST-trained weights: the maintainers request proof of a NIST license for those models and separately link open MassSpecGym-trained weights. Prefer the latter for an initial reproducible experiment; audit their pretraining overlap and artifact terms. The DreaMS Zenodo page exposed a license heading without readable license text in this research session, so checkpoint redistribution rights remain unverified.

## Measured baseline results (2026-09-23)

All rows below use a **closed-universe structure-only candidate database**. Held-out spectral identities are removed globally. These are development diagnostics, not hidden-test or external candidate-coverage estimates. Random/scaffold/NP splits have 128 query molecules each; the novelty stress test has 64. The NP domain split is now used for comparison, not an untouched final holdout.

| Method | Split | MRR@25 | R@1 | R@5 | R@10 | R@25 |
|---|---|---:|---:|---:|---:|---:|
| mass | random | 0.1330 | 0.0391 | 0.2344 | 0.3359 | 0.5625 |
| cosine | random | 0.0221 | 0.0000 | 0.0312 | 0.0781 | 0.1875 |
| modified | random | 0.0224 | 0.0000 | 0.0391 | 0.0781 | 0.1875 |
| mass | scaffold | 0.1272 | 0.0312 | 0.2500 | 0.3750 | 0.6016 |
| cosine | scaffold | 0.0194 | 0.0000 | 0.0234 | 0.0781 | 0.2188 |
| modified | scaffold | 0.0193 | 0.0000 | 0.0234 | 0.0781 | 0.2188 |
| mass | np_domain | 0.3880 | 0.2344 | 0.5703 | 0.7734 | 0.9062 |
| cosine | np_domain | 0.1676 | 0.0625 | 0.2656 | 0.4453 | 0.6641 |
| modified | np_domain | 0.1684 | 0.0547 | 0.2734 | 0.4609 | 0.7188 |
| fingerprint_ridge | random | 0.1848 | 0.1094 | 0.2500 | 0.3516 | 0.5625 |
| fingerprint_ridge | scaffold | 0.1605 | 0.0703 | 0.2578 | 0.3516 | 0.6250 |
| fingerprint_ridge | np_domain | 0.2560 | 0.1250 | 0.4062 | 0.6016 | 0.7344 |
| gated_mass | random | 0.1330 | 0.0391 | 0.2344 | 0.3359 | 0.5625 |
| gated_mass | scaffold | 0.1272 | 0.0312 | 0.2500 | 0.3750 | 0.6016 |
| gated_mass | np_domain | 0.3880 | 0.2344 | 0.5703 | 0.7734 | 0.9062 |
| mass | novelty_0.4 | 0.4226 | 0.2656 | 0.6250 | 0.7812 | 0.9219 |
| fingerprint_ridge | novelty_0.4 | 0.2449 | 0.0938 | 0.3906 | 0.6562 | 0.8125 |

Candidate oracle recall is 1.0 for these closed-universe mass-window experiments; train-only candidate recall and MRR are zero for every strict held-out split. Mass-order R@100 is 0.9453 / 0.9531 / 0.9922 for random / scaffold / NP; R@1000 is 1.0. Median pool sizes are 86.5 / 75 / 13. This small NP isomer pool partly explains the strong mass-only result; expanded external databases will change that result and require revalidation.

The ridge baseline uses one seeded spectrum per each of 40,000 eligible timsTOF training identities, 2 Da fragment/loss bins and mode/precursor metadata, predicting 2048-bit Morgan fingerprints. Fixed alpha=20; no alpha search. Fit time was approximately 2–2.4 seconds per fold on the local CPU after preprocessing; reported inference experiment times also include query assembly and comparisons. No deep transformer was trained.

Scaffold separation does not guarantee distant chemistry: among 128 scaffold queries, 17 still have a training neighbor with Tanimoto >=0.7 and only one has max similarity <=0.4. The additional 64-query natural-product stress split removes **8,671** structures with similarity >0.4 to any query, verifies the remaining maximum <=0.4, and saves its exclusion manifest. It exposes the ridge model's weak transfer to distant natural products.

Selected default: near-exact aggregate spectral cosine >=0.999 retains spectral ordering; otherwise mass ordering. This matches mass-only scores on the strict holdouts and preserves the visible placeholder exact-match behavior. The fixed threshold has **not** been calibrated on independently measured library-known compounds; the hidden competition mix is unknown. Plain cosine and learned hybrid configurations remain available. The learned comparator was not promoted because it regressed on the NP domain and distant-chemistry tests.

### Artifact and execution status

- Chemistry: all 277,566 input SMILES parsed; 274,195 canonical candidates under RDKit 2026.03.3. Canonical keys differ from supplied keys on 71,341 rows. No disconnected salts were present in the resulting structures.
- Adduct parser handles every observed training annotation except `[Cat]2+` (15 rows). These rows remain in the library but that unsupported adduct is not converted silently. All visible test adducts are supported.
- enveda-180 structure-derived absolute mass errors: median 1.69 ppm, 95th percentile 3.56 ppm, all within 10 ppm. NP examples: median 1.68 ppm, 95th percentile 3.74 ppm, 99.83% within 10 ppm. Source-specific details are in `candidate_audit.json`.
- The whole test pipeline was executed from the actual notebook code with offline local path overrides. It writes the Kaggle path `/kaggle/working/submission.csv` by default. Runtime is measured locally, not on Kaggle hardware.
- The local output contains all 400 visible IDs once, 2–25 candidates each (mean 23.265), with valid RDKit SMILES and no duplicate canonical identities. No query needed the out-of-window fallback. Final runtime is in `../submission.stats.json`.
- The bundle is approximately 1.53 GB, with source, configs, full candidate/library artifacts, optional ridge weights, Python 3.11/3.12 offline RDKit wheels, provenance and SHA-256 manifest. Standard Kaggle NumPy/pandas/PyArrow/SciPy are assumed; tested local versions are recorded in `environment.json`. No network is used by inference.
- Six sanity tests cover adducts, equivalence, metric rank boundaries/invalid predictions, duplicate removal, grouping, matching and submission constraints. Additional standalone notebook execution exercises the real 400-molecule pipeline.

### What remains beyond this first milestone

External candidate ingestion and its independent coverage measurement; an independent measured-library-known holdout; stronger learned models and externally pretrained embeddings; formula prediction; spectrum simulation; MRR-calibrated ensembles; de novo candidates; and a real Kaggle rerun/score remain unperformed. No external dataset/model or score improvement is fabricated. The next implementation should freeze a COCONUT/LOTUS snapshot and rerun coverage/ranking before expanding model complexity.
