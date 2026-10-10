"""Blind second review of the manual QC artifact labels.

1. ``python scripts/qc_verify.py qc.verify.mode=make``
   Picks ``qc.verify.n`` images from the labeled ``review_sample.csv`` (enriched with images
   labeled as having an artifact, then shuffled), renders large numbered tiles WITHOUT the
   first reviewer's labels (``verify_XX.png``) and writes a blank ``verify_sample.csv``.
2. The second reviewer fills ``edge_strip`` and ``horizontal_line`` (0/1) in
   ``verify_sample.csv`` by looking only at the ``verify_XX.png`` sheets.
   Labeling criteria (same for both reviewers):

   * ``edge_strip = 1``: a straight, bright (near-white) band along any image edge that is
     not tissue, i.e. a leftover scanner strip, including a short bar along the top edge.
     The pectoral muscle (a wide bright triangle) and ragged speckle on the skin line do
     not count.
   * ``horizontal_line = 1``: a straight, near-horizontal line or edge running across the
     breast or pectoral region (compression paddle or film edge). Curved tissue boundaries
     do not count.

   Don't open ``review_sample.csv`` before scoring; it holds the first reviewer's labels.
3. ``python scripts/qc_verify.py qc.verify.mode=score``
   Compares both reviewers' labels (agreement, Cohen's kappa, disagreements) and saves the
   result to ``verify_result.json`` and ``reports/qc_label_verification.json``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2
import hydra
import pandas as pd
from omegaconf import DictConfig

from breast_cancer.data.pipeline import read_png
from breast_cancer.data.qc import (
    contact_sheet,
    render_tile,
    score_verification,
    select_verification_sample,
)

log = logging.getLogger(__name__)


def make(cfg: DictConfig, qc_dir: Path, copy_dir: Path) -> None:
    """Write the blind verification sheets and blank label file."""
    target = qc_dir / "verify_sample.csv"
    if target.exists():
        raise FileExistsError(f"{target} exists (may hold your labels); delete it to resample")
    review = pd.read_csv(qc_dir / "review_sample.csv")
    v = cfg.qc.verify
    sample = select_verification_sample(review, v.n, v.max_positive_frac, cfg.seed)
    sample.to_csv(target, index=False)
    per_sheet = v.cols * v.rows
    for s, start in enumerate(range(0, len(sample), per_sheet)):
        rows = sample.iloc[start : start + per_sheet]
        tiles = [
            render_tile(
                read_png(copy_dir / "images" / f"{r.image_id}.png"),
                [],
                f"#{r.k}",
                (v.tile_w, v.tile_h),
            )
            for r in rows.itertuples()
        ]
        cv2.imwrite(str(qc_dir / f"verify_{s:02d}.png"), contact_sheet(tiles, v.cols, v.rows))
    log.info("wrote %d images to %s and verify_XX.png sheets", len(sample), target)


def score(cfg: DictConfig, qc_dir: Path) -> None:
    """Compare the second review with the first and save the agreement report."""
    first = pd.read_csv(qc_dir / "review_sample.csv")
    second = pd.read_csv(qc_dir / "verify_sample.csv", dtype=str, keep_default_na=False)
    result = score_verification(first, second)
    (qc_dir / "verify_result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    report = Path(cfg.paths.reports) / "qc_label_verification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(qc_dir / "verify_result.json", report)
    log.info("verification result: %s", json.dumps(result, indent=1))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the ``make`` or ``score`` step of the blind QC label verification."""
    copy_dir = Path(cfg.data.processed_dir) / cfg.qc.subdir
    qc_dir = Path(cfg.data.processed_dir) / "qc" / cfg.qc.subdir
    if cfg.qc.verify.mode == "make":
        make(cfg, qc_dir, copy_dir)
    elif cfg.qc.verify.mode == "score":
        score(cfg, qc_dir)
    else:
        raise ValueError(f"qc.verify.mode must be make or score, got {cfg.qc.verify.mode!r}")


if __name__ == "__main__":
    main()
