"""Automatic artifact detector for preprocessed mammograms (used by the QC script).

Detects a saturated scanner strip that survived cropping: a tall, thin component of
near-maximum pixels inside a narrow band along an image edge that ends before the band's
inner boundary. That last condition separates strips from the pectoral muscle, which is
also bright and runs along the chest-wall edge in MLO views but is much wider than the band.

Horizontal paddle/film lines are not detected automatically (they sit inside the bright
pectoral region and are hard to separate from tissue edges); QC estimates their prevalence
by manual review of a random sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
