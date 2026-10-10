# Claude Code prompts — paste one phase at a time

## Phase 1 — Scaffold
Scaffold the repository exactly as described in CLAUDE.md: git init, pyproject.toml with the listed dependencies, the src/breast_cancer package layout with empty modules, Hydra config skeleton, Makefile (setup, lint, test, train), .gitignore that excludes data/, pre-commit with ruff and black, and a GitHub Actions workflow that runs lint + pytest. Don't write model code yet.

## Phase 1 — Data pipeline
Write src/breast_cancer/data/dicom_io.py and preprocessing.py: read DICOMs with pydicom, apply windowing/VOI LUT, normalize, detect and crop the breast region (Otsu threshold + largest contour), flip to canonical orientation, resize keeping aspect ratio to a configurable long side, and save as PNG with a metadata CSV (patient_id, laterality, view, label, pathology, density, original path). Then scripts/preprocess.py driven by Hydra. Add unit tests using a synthetic image.

## Phase 1 — Splits
Write src/breast_cancer/data/splits.py that creates stratified, patient-level train/val/test splits from the metadata CSV with a fixed seed, saved to a CSV. Add a pytest that fails if any patient_id appears in more than one split.

## Phase 1 — Baseline
Implement a single-image baseline: timm EfficientNet-B0, whole image at 512px, BCE loss, AMP, W&B logging, saving best checkpoint by val AUC. Write evaluation/metrics.py with AUC + bootstrap 95% CI and sensitivity at 90% specificity.

## Phase 2 — Patch classifier
Using CBIS-DDSM ROI masks, write a patch dataset that samples lesion-centered and random background patches. Train a patch classifier (benign / malignant / background). Then build a whole-image model initialized from the patch backbone, fine-tuned at ≥1024px with gradient accumulation. Compare against the baseline in a results table.

## Phase 3 — Multi-view fusion
Build a dataset grouping images by patient + laterality into CC/MLO pairs (handle missing views). Implement a fusion model: shared backbone, per-view pooled features, concatenation or attention fusion, per-breast output. Optionally add a contralateral comparison branch. Compare with the single-view model.

## Phase 3 — Uncertainty & explainability
Add MC Dropout inference and a deep-ensemble option. Output mean probability + predictive entropy. Add Grad-CAM for the fusion model. Produce a referral curve: performance when the most uncertain X% of cases are deferred.

## Phase 4 — External validation & analysis
Write a VinDr-Mammo preprocessing config reusing the same pipeline. Evaluate the final model once on VinDr. Report AUC with CIs per breast density (A–D), a reliability diagram + ECE, and apply temperature scaling fitted on the CBIS validation set. Save figures to reports/figures/.

## Phase 5 — Deployment
Export the final model to ONNX and verify outputs match PyTorch within tolerance (add a test). Build a FastAPI app with /predict accepting DICOM or PNG for up to 4 views, returning probability, uncertainty, and a base64 heatmap. Dockerize it. Build a Gradio demo in app/ for Hugging Face Spaces.

## Phase 6 — Documentation
Write a README with: demo link at the top, problem statement, mermaid architecture diagram, datasets and how to obtain them, results tables with CIs (internal vs external), subgroup and calibration figures, an ablation table, reproduction steps, limitations, and a "not for clinical use" disclaimer. Also write MODEL_CARD.md.
