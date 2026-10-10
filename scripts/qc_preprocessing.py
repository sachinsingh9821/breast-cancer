"""Quality-check a preprocessed copy.

Usage:
    python scripts/qc_preprocessing.py data=cbis_ddsm [qc.subdir=master_2048]

Writes to ``<processed_dir>/qc/<subdir>/``:

* ``summary.json``: geometry/mask statistics and the automatic edge-strip rate.
* ``flags.csv``: automatic edge-strip flag per image.
* ``masks_XX.png``: images whose lesion masks needed attention, with mask outlines.
* ``review_XX.png`` + ``review_sample.csv``: a fixed random sample (numbered tiles) for
  manual artifact review; fill in the ``edge_strip`` / ``horizontal_line`` columns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from breast_cancer.data.dicom_io import normalize_intensity
from breast_cancer.data.pipeline import read_png
from breast_cancer.data.qc import QCParams, detect_edge_strip

log = logging.getLogger(__name__)


def summarize(meta: pd.DataFrame, lesions: pd.DataFrame) -> dict[str, object]:
    """Summary statistics of a preprocessed copy."""
    agree = meta["chest_wall_side"].str[0].str.upper() == meta["laterality"].str[0]
    return {
        "n_images": len(meta),
        "n_lesions": len(lesions),
        "flipped_frac": round(float(meta["flipped"].mean()), 3),
        "chest_wall_side_matches_laterality_frac": round(float(agree.mean()), 3),
        "mask_status": lesions["mask_status"].value_counts().to_dict(),
        "lesions_with_empty_mask": int((lesions["mask_area"] == 0).sum()),
        "crop_area_frac": meta.eval("(y1 - y0) * (x1 - x0) / (orig_h * orig_w)")
        .describe()
        .round(3)
        .to_dict(),
        "out_aspect_w_over_h": (meta["out_w"] / meta["out_h"]).describe().round(3).to_dict(),
    }


def tile(img: np.ndarray, masks: list[np.ndarray], label: str, size: tuple[int, int]) -> np.ndarray:
    """Render one image (percentile-windowed) with red lesion outlines, letterboxed to ``size``."""
    view = (normalize_intensity(img, 0.5, 99.5, mask=img > 0) * 255).astype(np.uint8)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    thickness = max(2, max(img.shape) // 300)
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 0, 255), thickness)
    w, h = size
    scale = min(w / view.shape[1], h / view.shape[0])
    view = cv2.resize(view, (int(view.shape[1] * scale), int(view.shape[0] * scale)))
    canvas = np.full((h, w, 3), 64, np.uint8)
    canvas[: view.shape[0], : view.shape[1]] = view
    cv2.putText(canvas, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    return canvas


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Write QC stats and contact sheets for one preprocessed copy."""
    copy_dir = Path(cfg.data.processed_dir) / cfg.qc.subdir
    out = Path(cfg.data.processed_dir) / "qc" / cfg.qc.subdir
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(copy_dir / "metadata.csv")
    lesions = pd.read_csv(copy_dir / "lesions.csv", keep_default_na=False)

    summary = summarize(meta, lesions)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("summary: %s", json.dumps(summary, indent=1))

    params = QCParams(**cfg.qc.detector)
    flags = meta[["image_id", "view"]].copy()
    flags["edge_strip_auto"] = [
        detect_edge_strip(read_png(copy_dir / "images" / f"{i}.png"), params)
        for i in meta["image_id"]
    ]
    flags.to_csv(out / "flags.csv", index=False)
    summary["edge_strip_auto_frac"] = round(float(flags["edge_strip_auto"].mean()), 4)
    summary["edge_strip_auto_frac_by_view"] = (
        flags.groupby("view")["edge_strip_auto"].mean().round(4).to_dict()
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("automatic edge-strip flags: %d/%d", int(flags["edge_strip_auto"].sum()), len(flags))

    masks_of = {k: [p for p in g["mask_path"] if p] for k, g in lesions.groupby("image_id")}
    size = (cfg.qc.tile_w, cfg.qc.tile_h)

    def write_sheets(ids: list[str], prefix: str, outlines: bool) -> None:
        """Render ``ids`` as numbered tiles into ``<prefix>_XX.png`` contact sheets."""
        per_sheet = cfg.qc.cols * cfg.qc.rows
        for s, start in enumerate(range(0, len(ids), per_sheet)):
            tiles = []
            for k, image_id in enumerate(ids[start : start + per_sheet], start=start):
                img = read_png(copy_dir / "images" / f"{image_id}.png")
                masks = (
                    [read_png(copy_dir / p) for p in masks_of.get(image_id, [])] if outlines else []
                )
                tiles.append(tile(img, masks, f"#{k} {image_id}", size))
            tiles += [np.full((size[1], size[0], 3), 32, np.uint8)] * (per_sheet - len(tiles))
            rows = [
                np.hstack(tiles[r * cfg.qc.cols : (r + 1) * cfg.qc.cols])
                for r in range(cfg.qc.rows)
            ]
            cv2.imwrite(str(out / f"{prefix}_{s:02d}.png"), np.vstack(rows))

    # Lesions whose mask needed attention, with outlines.
    mask_flagged = lesions.loc[lesions["mask_status"] != "ok", "image_id"].unique().tolist()
    write_sheets(mask_flagged, "masks", outlines=True)
    # Fixed random sample for manual artifact review (no outlines, so edges stay visible).
    sample = meta["image_id"].sample(min(cfg.qc.review_n, len(meta)), random_state=cfg.seed)
    review = flags.set_index("image_id").loc[sample].reset_index()
    review.insert(0, "k", range(len(review)))
    review["edge_strip"] = ""  # filled in by the reviewer
    review["horizontal_line"] = ""
    review.to_csv(out / "review_sample.csv", index=False)
    write_sheets(review["image_id"].tolist(), "review", outlines=False)
    log.info("wrote mask and review sheets to %s", out)


if __name__ == "__main__":
    main()
