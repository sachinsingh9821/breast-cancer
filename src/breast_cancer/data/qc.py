"""Automatic artifact detector for preprocessed mammograms (used by the QC script).

Detects a saturated scanner strip that survived cropping: a tall, thin component of
near-maximum pixels inside a narrow band along an image edge that ends before the band's
inner boundary. That last condition separates strips from the pectoral muscle, which is
also bright and runs along the chest-wall edge in MLO views but is much wider than the band.

Horizontal paddle/film lines are not detected automatically (they sit inside the bright
pectoral region and are hard to separate from tissue edges); QC estimates their prevalence
by manual review of a random sample, and :func:`select_verification_sample` /
:func:`score_verification` support a blind second review of those manual labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from breast_cancer.data.dicom_io import normalize_intensity


@dataclass(frozen=True)
class QCParams:
    """Detector thresholds (all come from the Hydra ``qc`` config)."""

    work_long_side: int  # detectors run on a copy downscaled to this long side
    strip_band_frac: float  # width of the edge band searched for strips
    strip_level_frac: float  # pixel >= this fraction of the image max counts as saturated
    strip_min_frac: float  # strip must span this fraction of the edge length


def _downscale(img: np.ndarray, long_side: int) -> np.ndarray:
    """Downscale (never upscale) so the long side is at most ``long_side``."""
    scale = long_side / max(img.shape)
    if scale >= 1:
        return img
    size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def detect_edge_strip(img: np.ndarray, p: QCParams) -> bool:
    """Whether a saturated strip runs along any edge of a preprocessed image."""
    small = _downscale(img, p.work_long_side).astype(np.float32)
    if small.max() <= 0:
        return False
    bright = (small >= p.strip_level_frac * small.max()).astype(np.uint8)
    h, w = bright.shape
    bx, by = max(1, round(w * p.strip_band_frac)), max(1, round(h * p.strip_band_frac))
    # Left/right bands are transposed so every band is "edge at column 0, length along rows".
    bands = [
        bright[:, :bx],
        bright[:, w - bx :][:, ::-1],
        bright[:by, :].T,
        bright[h - by :, :][::-1, :].T,
    ]
    for band in bands:
        n, _, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(band), 8)
        for x, _, cw, ch, _ in stats[1:]:
            ends_inside_band = x + cw < band.shape[1]  # pectoral muscle fills the whole band
            if ends_inside_band and ch >= p.strip_min_frac * band.shape[0]:
                return True
    return False


ARTIFACT_COLUMNS = ("edge_strip", "horizontal_line")


def select_verification_sample(
    review: pd.DataFrame, n: int, max_positive_frac: float, seed: int
) -> pd.DataFrame:
    """Pick a blind verification subset of a labeled review sample.

    Images labeled with any artifact are over-sampled (up to ``max_positive_frac`` of ``n``)
    so agreement is not dominated by easy clean images; the rest are random clean images.
    The result is shuffled and carries no labels.

    Args:
        review: Review sample with 0/1 ``edge_strip`` and ``horizontal_line`` columns.
        n: Subset size.
        max_positive_frac: Maximum fraction of the subset drawn from labeled positives.
        seed: Random seed.

    Returns:
        ``k, image_id`` plus empty artifact columns for the second reviewer.
    """
    raw = review[list(ARTIFACT_COLUMNS)]
    if raw.isna().any().any() or (raw.astype(str) == "").any().any():
        raise ValueError("review sample is not fully labeled (edge_strip / horizontal_line)")
    labels = raw.astype(int)
    positive = review[labels.any(axis=1)]
    negative = review[~labels.any(axis=1)]
    n_pos = min(len(positive), int(n * max_positive_frac))
    n_neg = min(len(negative), n - n_pos)
    picked = pd.concat(
        [positive.sample(n_pos, random_state=seed), negative.sample(n_neg, random_state=seed)]
    ).sample(frac=1, random_state=seed)
    out = picked[["image_id"]].reset_index(drop=True)
    out.insert(0, "k", range(len(out)))
    for col in ARTIFACT_COLUMNS:
        out[col] = ""
    return out


def score_verification(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, object]:
    """Agreement between two reviewers' 0/1 artifact labels on the same images.

    Args:
        first: Labels from the first reviewer (``image_id`` + artifact columns).
        second: Labels from the second reviewer on a subset of the same images.

    Returns:
        Per-artifact counts, percent agreement, Cohen's kappa and the disagreeing image IDs.
    """
    cols = list(ARTIFACT_COLUMNS)
    if second[cols].isna().any().any() or (second[cols].astype(str) == "").any().any():
        raise ValueError("verification sheet has unlabeled rows")
    merged = second.merge(first, on="image_id", suffixes=("_second", "_first"))
    if len(merged) != len(second):
        raise ValueError("verification images missing from the first reviewer's labels")
    result: dict[str, object] = {"n_images": len(merged)}
    for col in cols:
        a = merged[f"{col}_first"].astype(int)
        b = merged[f"{col}_second"].astype(int)
        both_constant = a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]
        result[col] = {
            "positives_first": int(a.sum()),
            "positives_second": int(b.sum()),
            "agreement": round(float((a == b).mean()), 3),
            "cohen_kappa": 1.0 if both_constant else round(float(cohen_kappa_score(a, b)), 3),
            "disagreements": merged.loc[a != b, "image_id"].tolist(),
        }
    return result


def render_tile(
    img: np.ndarray, masks: list[np.ndarray], label: str, size: tuple[int, int]
) -> np.ndarray:
    """Render an image (percentile-windowed, red lesion outlines) letterboxed to ``size``.

    Args:
        img: Preprocessed image (0 = background).
        masks: 8-bit lesion masks to outline (may be empty).
        label: Text drawn in the top-left corner.
        size: ``(width, height)`` of the tile.

    Returns:
        BGR ``uint8`` tile.
    """
    view = (normalize_intensity(img, 0.5, 99.5, mask=img > 0) * 255).astype(np.uint8)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    thickness = max(2, max(img.shape) // 300)
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 0, 255), thickness)
    w, h = size
    scale = min(w / view.shape[1], h / view.shape[0])
    view = cv2.resize(
        view, (int(view.shape[1] * scale), int(view.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((h, w, 3), 64, np.uint8)
    canvas[: view.shape[0], : view.shape[1]] = view
    font_scale = max(0.33, w / 600)
    cv2.putText(canvas, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
    return canvas


def contact_sheet(tiles: list[np.ndarray], cols: int, rows: int) -> np.ndarray:
    """Arrange equally sized tiles into a ``rows`` x ``cols`` grid, padding with dark tiles."""
    blank = np.full_like(tiles[0], 32)
    tiles = tiles + [blank] * (cols * rows - len(tiles))
    return np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
