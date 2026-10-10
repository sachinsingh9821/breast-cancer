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
import pandas as pd
from omegaconf import DictConfig

from breast_cancer.data.pipeline import read_png
from breast_cancer.data.qc import QCParams, contact_sheet, detect_edge_strip, render_tile

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
                tiles.append(render_tile(img, masks, f"#{k} {image_id}", size))
            sheet = contact_sheet(tiles, cfg.qc.cols, cfg.qc.rows)
            cv2.imwrite(str(out / f"{prefix}_{s:02d}.png"), sheet)

    # Lesions whose mask needed attention, with outlines.
    mask_flagged = lesions.loc[lesions["mask_status"] != "ok", "image_id"].unique().tolist()
    write_sheets(mask_flagged, "masks", outlines=True)
    # Fixed random sample for manual artifact review (no outlines, so edges stay visible).
    # Never overwrite a sample that may already hold manual labels.
    review_path = out / "review_sample.csv"
    if review_path.exists():
        log.warning("%s exists (may hold manual labels); not regenerating it", review_path)
    else:
        sample = meta["image_id"].sample(min(cfg.qc.review_n, len(meta)), random_state=cfg.seed)
        review = flags.set_index("image_id").loc[sample].reset_index()
        review.insert(0, "k", range(len(review)))
        review["edge_strip"] = ""  # filled in by the reviewer (0/1)
        review["horizontal_line"] = ""
        review.to_csv(review_path, index=False)
        write_sheets(review["image_id"].tolist(), "review", outlines=False)
    log.info("wrote mask and review sheets to %s", out)


if __name__ == "__main__":
    main()
