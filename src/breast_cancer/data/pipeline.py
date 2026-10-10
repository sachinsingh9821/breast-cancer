"""Batch preprocessing: DICOM -> master PNG copy -> derived training resolutions.

Layout of a preprocessed copy (``master_2048/`` or ``ls1024/`` etc.)::

    images/<image_id>.png         16-bit, breast cropped, chest wall on the canonical side
    masks/<abnormality_uid>.png   8-bit 0/255 lesion mask, pixel-aligned with its image
    records/<image_id>.json       per-image record (master only; marks the image as done)
    metadata.csv                  one row per image (labels + geometry)
    lesions.csv                   one row per lesion (labels + bounding box in this copy)

The master copy is built once from the DICOMs. Derived copies are made from the master only,
so the raw DICOMs can be deleted once the master is complete.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from breast_cancer.data.dicom_io import read_dicom_image
from breast_cancer.data.preprocessing import (
    CropParams,
    Side,
    apply_transform,
    preprocess_mammogram,
    resize_long_side,
)

log = logging.getLogger(__name__)

IMAGE_COLUMNS = [
    "image_id",
    "patient_id",
    "laterality",
    "view",
    "label",
    "pathology",
    "abnormality_types",
    "n_abnormalities",
    "breast_density",
    "assessment",
    "official_split",
    "series_uid",
    "full_file",
]
EMPTY_MASK = np.zeros((1, 1), dtype=np.uint8)
LESION_COLUMNS = [
    "abnormality_uid",
    "image_id",
    "patient_id",
    "abnormality_type",
    "abnormality_id",
    "pathology",
    "malignant",
    "assessment",
    "subtlety",
    "mass_shape",
    "mass_margins",
    "calc_type",
    "calc_distribution",
    "mask_file",
]


def write_png(path: Path, arr: np.ndarray) -> None:
    """Write a PNG atomically (temp file + rename), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), arr):
        raise OSError(f"failed to write {path}")
    os.replace(tmp, path)


def read_png(path: str | Path) -> np.ndarray:
    """Read a PNG keeping its bit depth (8 or 16-bit, single channel)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise OSError(f"failed to read {path}")
    return arr


def mask_bbox(mask: np.ndarray) -> dict[str, int]:
    """Bounding box (end-exclusive) and area of the nonzero pixels of a mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"bbox_y0": -1, "bbox_y1": -1, "bbox_x0": -1, "bbox_x1": -1, "mask_area": 0}
    return {
        "bbox_y0": int(ys.min()),
        "bbox_y1": int(ys.max()) + 1,
        "bbox_x0": int(xs.min()),
        "bbox_x1": int(xs.max()) + 1,
        "mask_area": int(ys.size),
    }


def process_image(
    image: dict[str, Any],
    lesions: list[dict[str, Any]],
    dicom_dir: str,
    out_dir: str,
    params: CropParams,
    long_side: int,
    canonical_side: Side,
) -> dict[str, Any]:
    """Preprocess one full mammogram and its lesion masks into the master copy.

    Args:
        image: Row of ``images.csv`` (needs ``image_id`` and ``full_file``).
        lesions: Rows of ``abnormalities.csv`` for this image (need ``abnormality_uid`` and
            ``mask_file``; a missing mask file is recorded, not fatal).
        dicom_dir: DICOM root.
        out_dir: Master copy root.
        params: Segmentation parameters.
        long_side: Output long side.
        canonical_side: Side the chest wall is moved to.

    Returns:
        The JSON record written to ``records/<image_id>.json``.
    """
    root, out = Path(dicom_dir), Path(out_dir)
    img = read_dicom_image(root / image["full_file"])
    processed, t, _ = preprocess_mammogram(img, params, long_side, canonical_side)
    write_png(out / "images" / f"{image['image_id']}.png", processed)

    lesion_records = []
    for les in lesions:
        rec: dict[str, Any] = {"abnormality_uid": les["abnormality_uid"]}
        if not isinstance(les.get("mask_file"), str):
            rec.update(mask_status="missing", **mask_bbox(EMPTY_MASK))
        else:
            mask = read_dicom_image(root / les["mask_file"])
            orig_area = int((mask > 0).sum())
            mask = (mask > 0).astype(np.uint8) * 255
            mapped = apply_transform(mask, t, cv2.INTER_NEAREST)
            write_png(out / "masks" / f"{les['abnormality_uid']}.png", mapped)
            box = mask_bbox(mapped)
            if mask.shape != img.shape:
                status = "resized"  # mask size differs from the image; resized to fit
            elif orig_area and not box["mask_area"]:
                status = "lost_in_crop"
            else:
                status = "ok"
            rec.update(mask_status=status, orig_mask_area=orig_area, **box)
        lesion_records.append(rec)

    record = {"image_id": image["image_id"], **t.to_dict(), "lesions": lesion_records}
    rec_path = out / "records" / f"{image['image_id']}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = rec_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, rec_path)  # written last: its presence means the image is complete
    return record


def build_master(
    images: pd.DataFrame,
    abnormalities: pd.DataFrame,
    dicom_dir: str | Path,
    out_dir: str | Path,
    params: CropParams,
    long_side: int,
    canonical_side: Side,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (or resume) the master copy for every available image.

    Images whose ``records/<image_id>.json`` exists are skipped. Afterwards ``metadata.csv``
    and ``lesions.csv`` are (re)written from all records.

    Returns:
        ``(metadata, lesions)``.
    """
    out = Path(out_dir)
    todo = images[images["available"]]
    todo = todo[~todo["image_id"].map(lambda i: (out / "records" / f"{i}.json").exists())]
    by_image = {k: g.to_dict("records") for k, g in abnormalities.groupby("image_id")}
    log.info(
        "master: %d images to process, %d already done",
        len(todo),
        images["available"].sum() - len(todo),
    )

    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_image,
                row,
                by_image.get(row["image_id"], []),
                str(dicom_dir),
                str(out),
                params,
                long_side,
                canonical_side,
            ): row["image_id"]
            for row in todo.to_dict("records")
        }
        for n, fut in enumerate(as_completed(futures), start=1):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - log and continue; rerun retries
                failures.append(futures[fut])
                log.error("%s failed: %s", futures[fut], exc)
            if n % 100 == 0 or n == len(futures):
                log.info("master: %d/%d", n, len(futures))
    if failures:
        log.error("%d images failed (rerun to retry): %s", len(failures), failures[:10])
    return write_master_tables(images, abnormalities, out)


def write_master_tables(
    images: pd.DataFrame, abnormalities: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble ``metadata.csv`` and ``lesions.csv`` of the master copy from its records."""
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((out_dir / "records").glob("*.json"))
    ]
    geometry = pd.DataFrame([{k: v for k, v in r.items() if k != "lesions"} for r in records])
    lesion_geo = pd.DataFrame([les for r in records for les in r["lesions"]])
    if geometry.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = images[IMAGE_COLUMNS].merge(geometry, on="image_id", how="inner")
    meta["image_path"] = "images/" + meta["image_id"] + ".png"
    lesions = abnormalities[LESION_COLUMNS].merge(lesion_geo, on="abnormality_uid", how="inner")
    lesions = lesions.rename(columns={"mask_file": "mask_dicom_file"})
    lesions["mask_path"] = np.where(
        lesions["mask_status"] != "missing", "masks/" + lesions["abnormality_uid"] + ".png", ""
    )
    meta.to_csv(out_dir / "metadata.csv", index=False)
    lesions.to_csv(out_dir / "lesions.csv", index=False)
    log.info(
        "master tables: %d images, %d lesions (mask status: %s)",
        len(meta),
        len(lesions),
        lesions["mask_status"].value_counts().to_dict(),
    )
    return meta, lesions


def _derive_one(
    master_dir: str, out_dir: str, image_id: str, mask_paths: list[str], long_side: int
) -> tuple[int, int]:
    """Downsample one master image and its masks; returns the new ``(h, w)``."""
    src, dst = Path(master_dir), Path(out_dir)
    img = read_png(src / "images" / f"{image_id}.png")
    small = resize_long_side(img, long_side, cv2.INTER_AREA)
    write_png(dst / "images" / f"{image_id}.png", small)
    for rel in mask_paths:
        m = read_png(src / rel)
        # area-average then keep any coverage, so tiny calcifications don't vanish
        m = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        write_png(dst / rel, ((m > 0) * 255).astype(np.uint8))
    return small.shape[0], small.shape[1]


def derive_copy(master_dir: str | Path, long_side: int, out_dir: str | Path, workers: int) -> None:
    """Make a lower-resolution copy of the master (images, masks and tables).

    Images already present in ``out_dir`` are skipped. Bounding boxes in ``lesions.csv`` are
    recomputed from the downsampled masks.
    """
    master_dir, out_dir = Path(master_dir), Path(out_dir)
    meta = pd.read_csv(master_dir / "metadata.csv")
    lesions = pd.read_csv(master_dir / "lesions.csv", keep_default_na=False)
    masks_of = {k: [p for p in g["mask_path"] if p] for k, g in lesions.groupby("image_id")}
    todo = [i for i in meta["image_id"] if not (out_dir / "images" / f"{i}.png").exists()]
    log.info("derive ls%d: %d images to process", long_side, len(todo))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _derive_one, str(master_dir), str(out_dir), i, masks_of.get(i, []), long_side
            )
            for i in todo
        ]
        for fut in futures:
            fut.result()

    sizes = {i: read_png(out_dir / "images" / f"{i}.png").shape for i in meta["image_id"]}
    meta = meta.copy()
    meta["out_h"] = meta["image_id"].map(lambda i: sizes[i][0])
    meta["out_w"] = meta["image_id"].map(lambda i: sizes[i][1])
    boxes = pd.DataFrame(
        [mask_bbox(read_png(out_dir / p) if p else EMPTY_MASK) for p in lesions["mask_path"]],
        columns=list(mask_bbox(EMPTY_MASK)),
    )
    lesions = pd.concat([lesions.drop(columns=boxes.columns), boxes], axis=1)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    lesions.to_csv(out_dir / "lesions.csv", index=False)
