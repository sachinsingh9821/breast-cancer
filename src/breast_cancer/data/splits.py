"""Patient-level train/val/test splits.

Policy for CBIS-DDSM (see CLAUDE.md, NOTES):

* **test** = the official CBIS-DDSM test set, kept for comparability with published work.
* Patients that appear in both the official train and test sets would leak; their official
  *train* images are **dropped** (recorded in a CSV), their test images stay in test.
* **train/val**: the remaining patients, split at the patient level, stratified by
  patient label (any malignant lesion) x abnormality type (mass / calc / both).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def _patient_abnormality_type(types: pd.Series) -> str:
    """Collapse per-image ``abnormality_types`` (e.g. ``calc+mass``) to mass/calc/both."""
    kinds = {k for t in types for k in str(t).split("+")}
    return "both" if len(kinds) > 1 else kinds.pop()


def patient_table(images: pd.DataFrame) -> pd.DataFrame:
    """Per-patient label (any malignant image) and abnormality type (mass/calc/both)."""
    return images.groupby("patient_id").agg(
        label=("label", "max"),
        abnormality_type=("abnormality_types", _patient_abnormality_type),
        n_images=("image_id", "nunique"),
    )


def overlap_patients(images: pd.DataFrame) -> pd.DataFrame:
    """Patients present in both the official train and test sets.

    Returns:
        One row per such patient with the number of official-train images that will be
        dropped and the number of test images kept.
    """
    both_img = images.groupby("patient_id")["in_both_official_splits"].any()
    splits = images.groupby("patient_id")["official_split"].nunique()
    ids = sorted(set(splits.index[splits > 1]) | set(both_img.index[both_img]))
    sub = images[images["patient_id"].isin(ids)]
    return (
        sub.groupby("patient_id")
        .agg(
            n_train_images_dropped=("official_split", lambda s: int((s == "train").sum())),
            n_test_images_kept=("official_split", lambda s: int((s == "test").sum())),
        )
        .reset_index()
    )


def _strata(patients: pd.DataFrame, stratify: Sequence[str], min_count: int) -> pd.Series:
    """Combine stratification columns into one stratum label per patient.

    Patients in strata smaller than ``min_count`` fall back to a label-only stratum, and if
    that is still too small, to a shared ``rare`` stratum, so the split never fails on tiny
    groups (e.g. the few patients with both mass and calc lesions).
    """
    strata = patients[list(stratify)].astype(str).agg("_".join, axis=1)
    fallbacks = [patients["label"].astype(str) + "_other", pd.Series("rare", index=strata.index)]
    for fallback in fallbacks:
        rare = strata.map(strata.value_counts()) < min_count
        if not rare.any():
            break
        log.info("merging %d patients from rare strata", int(rare.sum()))
        strata[rare] = fallback[rare]
    return strata


def make_splits(
    images: pd.DataFrame,
    val_frac: float,
    seed: int,
    stratify: Sequence[str] = ("label", "abnormality_type"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every image to train/val/test (or ``dropped``) with patient-level grouping.

    Args:
        images: Image table with ``image_id``, ``patient_id``, ``label``,
            ``abnormality_types``, ``official_split`` and ``in_both_official_splits``.
        val_frac: Fraction of eligible official-train patients assigned to validation.
        seed: Random seed for the train/val split.
        stratify: Patient-level columns to stratify on (from :func:`patient_table`).

    Returns:
        ``(splits, dropped)``: per-image ``image_id, patient_id, split, stratum`` and the
        overlap-patient table from :func:`overlap_patients`.
    """
    dropped = overlap_patients(images)
    overlap = set(dropped["patient_id"])
    is_test = images["official_split"] == "test"
    is_dropped = ~is_test & images["patient_id"].isin(overlap)
    pool = images[~is_test & ~is_dropped]

    patients = patient_table(pool)
    strata = _strata(patients, stratify, min_count=max(2, round(1 / val_frac)))
    stratifiable = strata.value_counts().min() >= 2
    train_ids, val_ids = train_test_split(
        patients.index.to_numpy(),
        test_size=val_frac,
        random_state=seed,
        stratify=strata if stratifiable else None,
    )
    split_of = {**{p: "train" for p in train_ids}, **{p: "val" for p in val_ids}}

    out = images[["image_id", "patient_id"]].copy()
    out["split"] = images["patient_id"].map(split_of)
    out.loc[is_test, "split"] = "test"
    out.loc[is_dropped, "split"] = "dropped"
    out["stratum"] = out["patient_id"].map(strata)
    check_no_patient_leakage(out)
    return out.reset_index(drop=True), dropped


def check_no_patient_leakage(splits: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any patient appears in more than one of train/val/test."""
    used = splits[splits["split"].isin(SPLITS)]
    n = used.groupby("patient_id")["split"].nunique()
    leaked = n.index[n > 1].tolist()
    if leaked:
        raise ValueError(f"{len(leaked)} patients in more than one split, e.g. {leaked[:5]}")
    if splits["split"].isna().any():
        raise ValueError("some images have no split")
