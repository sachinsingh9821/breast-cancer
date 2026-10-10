"""Index CBIS-DDSM: join case-description CSVs, the NBIA series manifest and DICOM headers.

How CSV paths map to files on disk:

* A CSV path looks like ``<Subject>/<StudyInstanceUID>/<SeriesInstanceUID>/00000N.dcm``.
* ``scripts/download_cbis_ddsm.py`` stores files as ``dicom/<SeriesInstanceUID>/1-N.dcm``.
* So the join key is the SeriesInstanceUID. The ``00000N`` file name is NOT usable: TCIA
  renames files, and the crop/mask index in the CSV is wrong for many rows. Crops and masks
  are told apart from the DICOM headers instead (a mask has the full image's size).

Outputs (written by :func:`build_index`):

* ``dicom_headers.csv``: one row per DICOM file (cached; only new series are scanned).
* ``abnormalities.csv``: one row per annotated lesion, with resolved full/mask/crop paths.
* ``images.csv``: one row per unique full mammogram. The same physical image sometimes
  appears twice (under a Mass and a Calc subject); identical pixels are merged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from breast_cancer.data.dicom_io import read_dicom_image, read_header

log = logging.getLogger(__name__)

CSV_FILES = {
    ("mass", "train"): "mass_case_description_train_set.csv",
    ("mass", "test"): "mass_case_description_test_set.csv",
    ("calc", "train"): "calc_case_description_train_set.csv",
    ("calc", "test"): "calc_case_description_test_set.csv",
}
COLUMN_MAP = {
    "patient_id": "patient_id",
    "breast_density": "breast_density",
    "breast density": "breast_density",  # calc CSVs use a space, mass CSVs an underscore
    "left or right breast": "laterality",
    "image view": "view",
    "abnormality id": "abnormality_id",
    "assessment": "assessment",
    "pathology": "pathology",
    "subtlety": "subtlety",
    "mass shape": "mass_shape",
    "mass margins": "mass_margins",
    "calc type": "calc_type",
    "calc distribution": "calc_distribution",
    "image file path": "image_csv_path",
    "cropped image file path": "crop_csv_path",
    "ROI mask file path": "mask_csv_path",
}
ROLE_BY_DESCRIPTION = {
    "full mammogram images": "full",
    "roi mask images": "mask",
    "cropped images": "crop",
}
HEADER_COLUMNS = [
    "series_uid",
    "file",
    "subject",
    "series_description",
    "role",
    "rows",
    "cols",
    "bits_stored",
    "photometric",
    "header_laterality",
]
PATHOLOGY_RANK = {"BENIGN_WITHOUT_CALLBACK": 0, "BENIGN": 1, "MALIGNANT": 2}


def load_case_descriptions(csv_dir: str | Path) -> pd.DataFrame:
    """Load and normalize the four case-description CSVs into one lesion-level table.

    Column names are unified (``breast density`` vs ``breast_density``), stray whitespace and
    newlines are stripped from paths, and the series UIDs are parsed out of the paths.

    Args:
        csv_dir: Directory containing the four ``*_case_description_*_set.csv`` files.

    Returns:
        One row per lesion with ``abnormality_type`` (mass/calc), ``official_split``
        (train/test), ``malignant`` and ``*_series_uid`` / ``*_subject`` columns.
    """
    frames = []
    for (kind, split), name in CSV_FILES.items():
        df = pd.read_csv(Path(csv_dir) / name)
        df.columns = [c.strip() for c in df.columns]
        unknown = set(df.columns) - set(COLUMN_MAP) - {"abnormality type"}
        if unknown:
            raise ValueError(f"{name}: unexpected columns {sorted(unknown)}")
        df = df.rename(columns=COLUMN_MAP).drop(columns=["abnormality type"])
        df["abnormality_type"] = kind
        df["official_split"] = split
        frames.append(df)
    cases = pd.concat(frames, ignore_index=True)

    for col in ("image", "crop", "mask"):
        path = cases[f"{col}_csv_path"].astype(str).str.strip()
        parts = path.str.split("/")
        if not (parts.str.len() == 4).all():
            raise ValueError(f"{col} paths are not <subject>/<study>/<series>/<file>")
        cases[f"{col}_csv_path"] = path
        cases[f"{col}_subject"] = parts.str[0]
        cases[f"{col}_series_uid"] = parts.str[2]
    cases["laterality"] = cases["laterality"].str.strip().str.upper()
    cases["view"] = cases["view"].str.strip().str.upper()
    cases["pathology"] = cases["pathology"].str.strip().str.upper()
    cases["malignant"] = (cases["pathology"] == "MALIGNANT").astype(int)
    cases["abnormality_uid"] = (
        cases["image_subject"] + "_" + cases["abnormality_id"].astype(str)
    )  # e.g. Mass-Training_P_00001_LEFT_CC_1, unique per lesion
    if cases["abnormality_uid"].duplicated().any():
        raise ValueError("abnormality_uid is not unique")
    return cases


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load the NBIA series manifest JSON written by the download script."""
    df = pd.DataFrame(json.loads(Path(path).read_text(encoding="utf-8")))
    return df.rename(
        columns={
            "SeriesInstanceUID": "series_uid",
            "StudyInstanceUID": "study_uid",
            "PatientID": "subject",
            "SeriesDescription": "series_description",
            "ImageCount": "image_count",
        }
    )[["series_uid", "study_uid", "subject", "series_description", "image_count"]]


def classify_role(description: str) -> str:
    """Map a DICOM ``SeriesDescription`` to ``full``, ``mask``, ``crop`` or ``unknown``."""
    return ROLE_BY_DESCRIPTION.get(str(description).strip().lower(), "unknown")


def _header_row(dicom_dir: Path, path: Path) -> dict[str, object]:
    """Read the header fields we index from one DICOM file."""
    ds = read_header(path)
    description = ds.get("SeriesDescription", "")
    return {
        "series_uid": path.parent.name,
        "file": path.relative_to(dicom_dir).as_posix(),
        "subject": ds.get("PatientID", ""),
        "series_description": description,
        "role": classify_role(description),
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
        "bits_stored": int(ds.get("BitsStored", 0)),
        "photometric": ds.get("PhotometricInterpretation", ""),
        "header_laterality": ds.get("Laterality", ds.get("ImageLaterality", "")),
    }


def scan_headers(
    dicom_dir: str | Path,
    series_uids: Iterable[str],
    cache: pd.DataFrame | None = None,
    workers: int = 8,
) -> pd.DataFrame:
    """Read headers of every DICOM in the given series directories.

    Series already present in ``cache`` are not re-read. Series directories that do not
    exist (not downloaded yet) are skipped.

    Args:
        dicom_dir: Root containing ``<SeriesInstanceUID>/*.dcm``.
        series_uids: Series to scan.
        cache: Previously scanned rows (same columns as the output).
        workers: Threads used to read headers.

    Returns:
        One row per DICOM file.
    """
    dicom_dir = Path(dicom_dir)
    done = set(cache["series_uid"]) if cache is not None and len(cache) else set()
    files = [
        f
        for uid in sorted(set(series_uids) - done)
        if (dicom_dir / uid).is_dir()
        for f in sorted((dicom_dir / uid).glob("*.dcm"))
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda f: _header_row(dicom_dir, f), files))
    new = pd.DataFrame(rows, columns=HEADER_COLUMNS)
    frames = [df for df in (cache, new) if df is not None and len(df)]
    return pd.concat(frames, ignore_index=True) if frames else new


def _pick_roi_files(
    roi_files: pd.DataFrame, full_shape: tuple[int, int] | None
) -> tuple[str | None, str | None, str]:
    """Choose the mask and crop file for one lesion from the files of its ROI series.

    The mask is the file with the full image's size (it is pixel-aligned with it); the header
    description is used only when sizes can't decide. Returns ``(mask, crop, how)``.
    """
    if roi_files.empty:
        return None, None, "missing"
    if full_shape is not None:
        same = roi_files[
            (roi_files["rows"] == full_shape[0]) & (roi_files["cols"] == full_shape[1])
        ]
        if len(same) == 1:
            other = roi_files.drop(same.index)
            crop = other["file"].iloc[0] if len(other) else None
            agrees = same["role"].iloc[0] == "mask"
            return same["file"].iloc[0], crop, "size" if agrees else "size_overrides_header"
    masks = roi_files[roi_files["role"] == "mask"]
    crops = roi_files[roi_files["role"] == "crop"]
    mask = masks["file"].iloc[0] if len(masks) == 1 else None
    crop = crops["file"].iloc[0] if len(crops) == 1 else None
    return mask, crop, "header" if mask else "ambiguous"


def resolve_files(cases: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    """Attach on-disk full image, mask and crop paths to each lesion.

    Args:
        cases: Output of :func:`load_case_descriptions`.
        headers: Output of :func:`scan_headers`.

    Returns:
        ``cases`` plus ``full_file``, ``mask_file``, ``crop_file`` (relative to the DICOM
        root, or NaN if not downloaded), ``rows``/``cols`` of the full image,
        ``header_laterality`` and ``mask_resolution`` (how the mask was identified).
    """
    by_series = {uid: g for uid, g in headers.groupby("series_uid")}
    out = []
    for row in cases.itertuples(index=False):
        rec = row._asdict()
        full = by_series.get(row.image_series_uid)
        if full is not None and len(full) == 1:
            f = full.iloc[0]
            rec.update(
                full_file=f["file"],
                rows=int(f["rows"]),
                cols=int(f["cols"]),
                header_laterality=f["header_laterality"],
            )
            shape: tuple[int, int] | None = (int(f["rows"]), int(f["cols"]))
        else:
            rec.update(full_file=None, rows=None, cols=None, header_laterality=None)
            shape = None
        roi_uids = {row.crop_series_uid, row.mask_series_uid}
        roi = [by_series[u] for u in roi_uids if u in by_series]
        downloaded = len(roi) == len(roi_uids)
        roi_files = pd.concat(roi) if downloaded else pd.DataFrame()
        mask, crop, how = _pick_roi_files(roi_files, shape)
        rec.update(mask_file=mask, crop_file=crop, mask_resolution=how)
        out.append(rec)
    return pd.DataFrame(out)


def pixel_hash(path: str | Path) -> str:
    """SHA-1 of an image's decoded pixel values (plus shape), for duplicate detection."""
    arr = read_dicom_image(path)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


def _worst_pathology(values: pd.Series) -> str:
    """Most severe pathology in ``values`` (MALIGNANT > BENIGN > BENIGN_WITHOUT_CALLBACK)."""
    return max(values, key=lambda v: PATHOLOGY_RANK.get(v, -1))


def build_image_table(
    abn: pd.DataFrame, dicom_dir: str | Path | None, hash_duplicates: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse lesions to one row per unique full mammogram.

    Full-image series that share patient/laterality/view are pixel-hashed; identical ones are
    merged into a single image. Distinct images of the same view (or unhashable ones, e.g.
    not yet downloaded) are kept separate with a suffix.

    Args:
        abn: Output of :func:`resolve_files`.
        dicom_dir: DICOM root used to hash candidate duplicates (None disables hashing).
        hash_duplicates: Whether to hash candidate duplicates at all.

    Returns:
        ``(images, abn)``: the image table and ``abn`` with an ``image_id`` column.
    """
    series = (
        abn.groupby("image_series_uid")
        .agg(
            patient_id=("patient_id", "first"),
            laterality=("laterality", "first"),
            view=("view", "first"),
            full_file=("full_file", "first"),
        )
        .reset_index()
    )
    series["pixel_hash"] = None
    key = ["patient_id", "laterality", "view"]
    dup = series.duplicated(key, keep=False)
    if hash_duplicates and dicom_dir is not None:
        for i in series.index[dup & series["full_file"].notna()]:
            series.at[i, "pixel_hash"] = pixel_hash(Path(dicom_dir) / series.at[i, "full_file"])
    # identical pixels -> same group; otherwise each series is its own group
    series["dedup_key"] = series["pixel_hash"].fillna(series["image_series_uid"])
    series = series.sort_values(key + ["image_series_uid"])
    groups = series.groupby(key + ["dedup_key"], sort=False).ngroup()
    series["group"] = groups.values
    base = series["patient_id"] + "_" + series["laterality"] + "_" + series["view"]
    rank = series.groupby(key)["group"].rank(method="dense").astype(int)
    n_distinct = series.groupby(key)["group"].transform("nunique")
    suffix = np.where(n_distinct > 1, "_" + (rank - 1).map(lambda r: "abcdefgh"[r]), "")
    series["image_id"] = base + suffix
    log.info(
        "full-image series: %d, candidate duplicates: %d, merged into %d images",
        len(series),
        int(dup.sum()),
        series["image_id"].nunique(),
    )

    abn = abn.merge(series[["image_series_uid", "image_id"]], on="image_series_uid", how="left")
    canonical = series.drop_duplicates("image_id")[["image_id", "image_series_uid", "full_file"]]
    agg = abn.groupby("image_id").agg(
        patient_id=("patient_id", "first"),
        laterality=("laterality", "first"),
        view=("view", "first"),
        label=("malignant", "max"),
        pathology=("pathology", _worst_pathology),
        abnormality_types=("abnormality_type", lambda s: "+".join(sorted(set(s)))),
        n_abnormalities=("abnormality_uid", "nunique"),
        breast_density=("breast_density", "max"),
        density_conflict=("breast_density", lambda s: s.nunique() > 1),
        assessment=("assessment", "max"),
        official_split=("official_split", lambda s: "test" if "test" in set(s) else "train"),
        in_both_official_splits=("official_split", lambda s: s.nunique() > 1),
        series_uids=("image_series_uid", lambda s: ";".join(sorted(set(s)))),
        rows=("rows", "first"),
        cols=("cols", "first"),
        header_laterality=("header_laterality", "first"),
    )
    images = canonical.merge(agg.reset_index(), on="image_id")
    images = images.rename(columns={"image_series_uid": "series_uid"})
    images["available"] = images["full_file"].notna()
    return images.sort_values("image_id").reset_index(drop=True), abn


def build_index(
    csv_dir: str | Path,
    manifest_path: str | Path,
    dicom_dir: str | Path,
    out_dir: str | Path,
    hash_duplicates: bool = True,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``dicom_headers.csv``, ``abnormalities.csv`` and ``images.csv`` in ``out_dir``.

    Safe to run on a partial download: missing series are reported and indexed as
    unavailable; rerun once the download is complete.

    Returns:
        ``(images, abnormalities)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_case_descriptions(csv_dir)
    manifest = load_manifest(manifest_path)
    needed = set(cases["image_series_uid"]) | set(cases["crop_series_uid"])
    needed |= set(cases["mask_series_uid"])
    missing_in_manifest = needed - set(manifest["series_uid"])
    if missing_in_manifest:
        raise ValueError(f"{len(missing_in_manifest)} CSV series are not in the NBIA manifest")

    cache_path = out_dir / "dicom_headers.csv"
    cache = pd.read_csv(cache_path) if cache_path.exists() else None
    headers = scan_headers(dicom_dir, needed, cache, workers)
    headers.to_csv(cache_path, index=False)
    log.info(
        "headers: %d files from %d/%d series",
        len(headers),
        headers.series_uid.nunique(),
        len(needed),
    )

    abn = resolve_files(cases, headers)
    images, abn = build_image_table(abn, dicom_dir, hash_duplicates)
    abn.to_csv(out_dir / "abnormalities.csv", index=False)
    images.to_csv(out_dir / "images.csv", index=False)
    log.info(
        "lesions: %d (mask found: %d; resolution: %s)",
        len(abn),
        int(abn["mask_file"].notna().sum()),
        abn["mask_resolution"].value_counts().to_dict(),
    )
    log.info("images: %d (available: %d)", len(images), int(images["available"].sum()))
    return images, abn
