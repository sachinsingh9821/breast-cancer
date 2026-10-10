"""Quality-check a preprocessed copy: summary stats and contact sheets with lesion outlines.

Usage:
    python scripts/qc_preprocessing.py data=cbis_ddsm [qc.subdir=master_2048] [qc.n_images=48]

Writes ``<processed_dir>/qc/<subdir>/summary.json`` and ``sheet_XX.png`` files.
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

    # random sample plus every lesion whose mask needed attention
    flagged = lesions.loc[lesions["mask_status"] != "ok", "image_id"].unique().tolist()
    sample = meta["image_id"].sample(min(cfg.qc.n_images, len(meta)), random_state=cfg.seed)
    ids = list(dict.fromkeys(flagged + sample.tolist()))
    masks_of = {k: [p for p in g["mask_path"] if p] for k, g in lesions.groupby("image_id")}
    per_sheet = cfg.qc.cols * cfg.qc.rows
    size = (cfg.qc.tile_w, cfg.qc.tile_h)
    for s, start in enumerate(range(0, len(ids), per_sheet)):
        tiles = []
        for image_id in ids[start : start + per_sheet]:
            img = read_png(copy_dir / "images" / f"{image_id}.png")
            masks = [read_png(copy_dir / p) for p in masks_of.get(image_id, [])]
            flag = "!" if image_id in flagged else ""
            tiles.append(tile(img, masks, f"{flag}{image_id}", size))
        tiles += [np.full((size[1], size[0], 3), 32, np.uint8)] * (per_sheet - len(tiles))
        rows = [
            np.hstack(tiles[r * cfg.qc.cols : (r + 1) * cfg.qc.cols]) for r in range(cfg.qc.rows)
        ]
        cv2.imwrite(str(out / f"sheet_{s:02d}.png"), np.vstack(rows))
    log.info("wrote %d contact sheets to %s", -(-len(ids) // per_sheet), out)


if __name__ == "__main__":
    main()
