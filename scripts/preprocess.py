"""CLI entry point: index, split and preprocess a dataset.

Stages (``preprocess.stages``), in order:

* ``index``  - join CSVs, series manifest and DICOM headers (``data.interim_dir``).
* ``splits`` - patient-level train/val/test splits (``data.processed_dir/splits.csv``).
* ``master`` - DICOM -> 16-bit PNG master copy at ``data.master_long_side`` (resumable).
* ``derive`` - cached lower-resolution copies for each of ``data.derived_long_sides``.

Usage:
    python scripts/preprocess.py data=cbis_ddsm
    python scripts/preprocess.py data=cbis_ddsm preprocess.stages=[index,splits]
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from breast_cancer.data.cbis_index import build_index
from breast_cancer.data.pipeline import build_master, derive_copy
from breast_cancer.data.preprocessing import CropParams
from breast_cancer.data.splits import make_splits
from breast_cancer.utils.repro import get_git_commit, set_seed

log = logging.getLogger(__name__)
STAGES = ("index", "splits", "master", "derive")


def run_index(cfg: DictConfig) -> None:
    """Build the CBIS-DDSM index tables."""
    d = cfg.data
    build_index(
        d.csv_dir,
        d.series_manifest,
        d.dicom_dir,
        d.interim_dir,
        d.index.hash_duplicates,
        d.num_workers,
    )


def run_splits(cfg: DictConfig) -> None:
    """Make patient-level splits and record dropped overlap patients."""
    d = cfg.data
    images = pd.read_csv(Path(d.interim_dir) / "images.csv")
    splits, dropped = make_splits(images, d.split.val_frac, d.split.seed, d.split.stratify)
    out = Path(d.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out / "splits.csv", index=False)
    dropped.to_csv(out / "dropped_overlap_patients.csv", index=False)
    report = Path(cfg.paths.reports) / f"{d.name}_dropped_overlap_patients.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "dropped_overlap_patients.csv", report)  # committed, cited in README
    per_patient = splits.drop_duplicates("patient_id")["split"].value_counts().to_dict()
    log.info("images per split: %s", splits["split"].value_counts().to_dict())
    log.info("patients per split: %s; overlap patients dropped: %d", per_patient, len(dropped))


def run_master(cfg: DictConfig) -> None:
    """Build (or resume) the master PNG copy from the DICOMs."""
    d = cfg.data
    interim = Path(d.interim_dir)
    images = pd.read_csv(interim / "images.csv")
    abn = pd.read_csv(interim / "abnormalities.csv")
    n_missing = int((~images["available"]).sum())
    if n_missing:
        log.warning("%d images not downloaded yet; they are skipped (rerun later)", n_missing)
    build_master(
        images,
        abn,
        d.dicom_dir,
        d.master_dir,
        CropParams(**d.crop),
        d.master_long_side,
        d.canonical_side,
        d.num_workers,
    )


def run_derive(cfg: DictConfig) -> None:
    """Make the cached lower-resolution copies from the master."""
    d = cfg.data
    for long_side in d.derived_long_sides:
        derive_copy(
            d.master_dir, long_side, Path(d.processed_dir) / f"ls{long_side}", d.num_workers
        )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the requested preprocessing stages with the composed Hydra config."""
    set_seed(cfg.seed)
    stages = list(cfg.preprocess.stages)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    if cfg.data.name != "cbis_ddsm":
        raise NotImplementedError(f"preprocessing for {cfg.data.name!r} is not implemented yet")

    run_info = {
        "git_commit": get_git_commit(cfg.paths.root),
        "stages": stages,
        "config": OmegaConf.to_container(cfg.data, resolve=True),
    }
    log.info("run: %s", json.dumps(run_info, indent=1))
    runners = {"index": run_index, "splits": run_splits, "master": run_master, "derive": run_derive}
    for stage in STAGES:
        if stage in stages:
            log.info("=== stage: %s ===", stage)
            runners[stage](cfg)
    out = Path(cfg.data.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_run.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
