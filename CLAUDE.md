# Breast Cancer — Multi-View Mammography Screening System

## What this project is
An end-to-end ML system that takes a 4-view screening mammogram exam (L-CC, L-MLO, R-CC, R-MLO), predicts malignancy per breast, localizes suspicious regions, and reports calibrated uncertainty. Built as a portfolio project for ML engineering roles. **Not for clinical use.**

## Goals (in priority order)
1. Honest evaluation: patient-level splits, external validation on a second dataset, bootstrap CIs.
2. Clean, reproducible engineering: configs, tests, experiment tracking, Docker.
3. Strong modeling: patch → whole-image transfer, multi-view fusion, uncertainty, explainability.
4. A live demo (Gradio on Hugging Face Spaces) and a polished README.

## Datasets
- **CBIS-DDSM** (TCIA) — primary training set. Has ROI masks and pathology labels.
- **VinDr-Mammo** (PhysioNet, credentialed) — external validation; has BI-RADS, density, bounding boxes.
- Raw data lives in `data/raw/` and is NEVER committed to git. Use DVC for tracking.

## Tech stack
- Python 3.11, PyTorch 2.x, timm, albumentations, pydicom, OpenCV
- Hydra for configs, Weights & Biases (or MLflow) for tracking, DVC for data
- pytest, ruff, black, pre-commit, GitHub Actions
- ONNX Runtime for inference, FastAPI for the API, Gradio for the demo, Docker

## Repository layout
```
configs/            # Hydra configs (data/, model/, train/, experiment/)
data/               # raw/, interim/, processed/  (gitignored, DVC-tracked)
src/breast_cancer/
  data/             # dicom io, breast cropping, datasets, splits
  models/           # backbones, patch classifier, multi-view fusion head
  training/         # train loops, losses, callbacks
  evaluation/       # metrics, bootstrap CIs, calibration, subgroup analysis
  explain/          # Grad-CAM
  uncertainty/      # MC dropout, ensembles
  serving/          # onnx export, FastAPI app
scripts/            # CLI entry points (preprocess.py, train.py, evaluate.py, export.py)
app/                # Gradio demo
tests/
notebooks/          # exploration ONLY — no logic that the pipeline depends on
reports/figures/    # plots used in the README
```

## Non-negotiable rules
- **Patient-level splits only.** No patient may appear in more than one of train/val/test. Write a test that asserts this.
- **Never touch the external test set** during model selection or tuning. It's evaluated once, at the end.
- Keep images at high resolution (≥1024 px long side for whole-image models). Don't downsample to 224.
- Preserve left/right orientation information; flip all images to a canonical side (e.g., breast on the left) and record it.
- All hyperparameters come from Hydra configs — no magic numbers in code.
- Set seeds and log the config + git commit hash with every run.
- Report AUC with 95% bootstrap CI, sensitivity at 90% specificity, and calibration (ECE + reliability plot).
- Every function in `src/` has type hints and a short docstring; core logic has unit tests.
- Don't add dependencies without updating `pyproject.toml`.

## Commands
- `make setup` (or `.\tasks.ps1 setup` on Windows PowerShell) — install deps + pre-commit hooks; same targets: lint, format, test, train
- `python scripts/preprocess.py data=cbis_ddsm` (stages: index, splits, master, derive; e.g. `preprocess.stages=[index,splits]`)
- `python scripts/train.py experiment=<name>`
- `python scripts/evaluate.py checkpoint=<path> data=vindr`
- `pytest -q` and `ruff check .` must pass before commit

## Hardware assumptions
Training on Kaggle/Colab GPUs (T4/P100, ~16 GB). Use mixed precision (AMP), gradient accumulation, and cached preprocessed PNG/NPY files instead of reading DICOMs during training.

## NOTES (carry into the README)
- **CBIS-DDSM label definition:** an image (or breast) is positive if it contains a malignant lesion, negative if it contains only benign lesions (BENIGN or BENIGN_WITHOUT_CALLBACK). CBIS-DDSM has **no normal images**: every image has at least one annotated lesion. So the CBIS task is malignant-vs-benign lesion classification, not cancer-vs-normal screening; VinDr-Mammo (which includes normal exams) is the screening-realistic evaluation.
- **CBIS-DDSM split:** the official test set is kept as the test set. 31 patients appear in both the official train and test sets; their official-train images are dropped (IDs listed in `reports/cbis_ddsm_dropped_overlap_patients.csv`). Validation is carved out of the remaining train patients at the patient level, stratified by patient label × abnormality type (mass/calc/both).
- **CBIS-DDSM quirks handled in the pipeline:** CSV file names (`00000N.dcm`) don't exist on disk (join on SeriesInstanceUID) and the crop/mask index is unreliable (masks identified by matching the full image's size); `breast density` vs `breast_density` column names; some images listed under both a Mass and a Calc subject (pixel-identical duplicates merged); image orientation doesn't follow laterality (flip decided from the image); saturated scanner strips and film labels (removed during cropping).
- **CBIS-DDSM laterality and orientation:** laterality, view and patient ID come from the TCIA subject ID (e.g. `Mass-Training_P_00001_LEFT_CC`). The CSV columns are checked against it (the index fails on any disagreement), and the DICOM `Laterality` header is kept only as `header_laterality` (a mismatch is logged). Labels agree across all three sources; what doesn't follow laterality is the *pixel orientation*. In an early sample, the chest wall was on the side expected from laterality in only 67% of images, so the canonical flip is decided from the image itself and recorded (`flipped`, `chest_wall_side`). Multi-view pairing must use `laterality`, never the flip state.
- **Data storage:** the master copy is 16-bit PNG at long side 2048 (`data/processed/cbis_ddsm/master_2048/`); raw DICOMs may be deleted after it is built. Training resolutions (1024, 512) are cached copies derived from the master.

## Current status
- [ ] Phase 1: scaffold, data pipeline, splits, baseline
  - [x] Scaffold (package, Hydra configs, tooling, CI, pre-commit)
  - [x] CBIS-DDSM downloader (`scripts/download_cbis_ddsm.py`, resumable) + the four case CSVs in `data/raw/cbis_ddsm/csv/`
  - [x] Data pipeline code, all tested on synthetic data and a fake CBIS download (69 tests):
        `dicom_io`, `preprocessing` (crop, strip/label removal, image-based flip, resize),
        `cbis_index` (CSV ↔ DICOM join by SeriesUID, masks by size, duplicate merge),
        `splits` (official test kept, 31 overlap patients dropped, stratified patient-level val),
        `pipeline` + `scripts/preprocess.py` (index → splits → master 2048 → derive 1024/512)
  - [x] QC tooling: `scripts/qc_preprocessing.py` (stats, automatic edge-strip flag, 300-image review sample)
        and `scripts/qc_verify.py` (blind second-reviewer check, agreement + Cohen's kappa)
  - [ ] **CBIS-DDSM download in progress** (as of 2026-10-10 19:06: 1986/6775 series, 38.8/152.3 GB, 25.5%).
        DICOMs go to `S:\cbis_ddsm\dicom` via a junction at `data/raw/cbis_ddsm/dicom`.
        Runs detached in a restart loop (4 workers): tcia_utils' threaded download crashed the
        interpreter twice (`Fatal Python error: PyEval_SaveThread` in ssl), and the loop resumes
        after a crash. Progress: `Get-Content data\raw\cbis_ddsm\download.log -Tail 5`;
        restarts: `data\raw\cbis_ddsm\download.restarts.log`; stop: `taskkill /PID (Get-Content data\raw\cbis_ddsm\download.pid) /T /F`.
  - [ ] Full pipeline run, QC, labeling, verification (next steps below)
  - [ ] Single-image baseline (EfficientNet-B0 @ 512) + `evaluation/metrics.py`
- [ ] Phase 2: patch classifier → whole-image model
- [ ] Phase 3: multi-view fusion, uncertainty, Grad-CAM
- [ ] Phase 4: external validation, density subgroups, calibration
- [ ] Phase 5: ONNX, FastAPI, Docker, Gradio demo, CI
- [ ] Phase 6: README, model card, figures

### Next steps (in order)
1. **Confirm the download finished**: `download.log` ends with "All 6775 series downloaded." If it reports failed series, rerun the downloader (it retries only those).
2. **Full pipeline run**: `python scripts/preprocess.py data=cbis_ddsm` (index with pixel-hash dedup of the 142 duplicate candidates → splits → master 2048 → derive 1024/512, about 10 min). Then check that `pytest -q` runs the real-splits leakage test (it is skipped until `data/processed/cbis_ddsm/splits.csv` exists), and commit `reports/cbis_ddsm_dropped_overlap_patients.csv`.
3. **QC run**: `python scripts/qc_preprocessing.py data=cbis_ddsm` → `data/processed/cbis_ddsm/qc/master_2048/`.
4. **Label the 300-image review sample** (Claude): fill `edge_strip` / `horizontal_line` (0/1) in `review_sample.csv` using the `review_XX.png` sheets and the criteria in `scripts/qc_verify.py`. Report each artifact's prevalence with a 95% CI and the automatic detector's precision/recall. If either artifact is above 5%, propose a fix (likely: lower `strip_min_frac` for top/bottom bands in the crop step, then rebuild master); otherwise document it in NOTES.
5. **Blind verification set for the owner**: `python scripts/qc_verify.py qc.verify.mode=make` → 30 images in `verify_00..02.png` + blank `verify_sample.csv`. The owner labels them without opening `review_sample.csv`, then `python scripts/qc_verify.py qc.verify.mode=score` → `reports/qc_label_verification.json`. Review the disagreements together.
6. Then: the single-image baseline (Phase 1, last item).
