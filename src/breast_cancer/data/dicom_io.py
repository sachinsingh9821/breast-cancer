"""DICOM reading and intensity normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut

UINT16_MAX = np.iinfo(np.uint16).max


def read_header(path: str | Path) -> pydicom.Dataset:
    """Read a DICOM header without loading pixel data."""
    return pydicom.dcmread(path, stop_before_pixels=True)


def read_dicom_image(path: str | Path, apply_voi: bool = True) -> np.ndarray:
    """Read a DICOM image as a ``uint16`` array where higher values mean denser tissue.

    MONOCHROME1 images are inverted. If the file carries a VOI LUT or window and
    ``apply_voi`` is set, it is applied and the result is rescaled to the full uint16 range;
    otherwise the stored values are kept unchanged, so no intensity information is lost.

    Args:
        path: Path to the DICOM file.
        apply_voi: Apply the file's VOI LUT / window when present.

    Returns:
        2-D ``uint16`` array.
    """
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    has_voi = "VOILUTSequence" in ds or "WindowCenter" in ds
    if apply_voi and has_voi:
        arr = apply_voi_lut(arr, ds).astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-9) * UINT16_MAX
    arr = np.asarray(arr)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    return np.clip(arr, 0, UINT16_MAX).astype(np.uint16)


def normalize_intensity(
    img: np.ndarray,
    low_pct: float,
    high_pct: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile-window an image to ``float32`` in [0, 1].

    Args:
        img: Input image of any numeric dtype.
        low_pct: Lower percentile mapped to 0.
        high_pct: Upper percentile mapped to 1.
        mask: Optional boolean mask; percentiles are computed only over ``True`` pixels.

    Returns:
        ``float32`` image clipped to [0, 1].
    """
    values = img[mask] if mask is not None and mask.any() else img
    lo, hi = (float(v) for v in np.percentile(values, [low_pct, high_pct]))
    out = (img.astype(np.float32) - np.float32(lo)) / np.float32(max(hi - lo, 1e-6))
    return np.clip(out, 0.0, 1.0)
