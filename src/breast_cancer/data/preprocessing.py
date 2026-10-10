"""Breast segmentation, cropping, canonical flipping and resizing.

A mammogram is turned into a model-ready image in two steps:

1. :func:`compute_transform` finds the breast (Otsu threshold + largest connected component,
   computed on a downscaled copy for speed), its bounding box, and which side the chest wall
   is on. The result is a :class:`Transform`, a small record that can be saved as metadata.
2. :func:`apply_transform` applies that record to the image, or to a lesion mask, so that
   images and masks stay pixel-aligned.

Scanned-film quirks handled here: saturated scanner strips along the image borders
(:func:`border_strip_mask`, plus a band of ``border_trim_frac`` ignored when segmenting),
film labels/markers (dropped by keeping only the largest component, and zeroed if they fall
inside the crop box), and orientation that does not follow laterality (the flip is decided
from the image, not the header).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from breast_cancer.data.dicom_io import normalize_intensity

Side = Literal["left", "right"]


@dataclass(frozen=True)
class CropParams:
    """Parameters for breast segmentation (all come from the Hydra data config)."""

    work_long_side: int  # segmentation runs on a copy downscaled to this long side
    high_pct: float  # percentile mapped to white before Otsu (robust to saturated pixels)
    border_trim_frac: float  # fraction of each border ignored when segmenting
    strip_scan_frac: float  # outer fraction of each side scanned for saturated scanner strips
    strip_min_frac: float  # a row/column is a strip if this fraction of it is near-saturated
    strip_level: int  # 8-bit level counted as near-saturated
    blur_ksize: int  # Gaussian blur kernel (odd), at work resolution
    morph_ksize: int  # opening/closing kernel, at work resolution
    margin_frac: float  # padding added around the bounding box, as a fraction of its size
    zero_background: bool  # zero pixels outside the (dilated) breast mask inside the crop
    background_dilate_px: int  # dilation of the breast mask before zeroing, at work resolution


@dataclass(frozen=True)
class Transform:
    """Geometric record mapping an original image to its preprocessed version.

    Coordinates refer to the original image: rows ``y0:y1`` and columns ``x0:x1`` are kept,
    then the crop is flipped horizontally if ``flipped``, then resized to ``out_h`` x ``out_w``.
    """

    orig_h: int
    orig_w: int
    y0: int
    y1: int
    x0: int
    x1: int
    flipped: bool
    chest_wall_side: Side
    out_h: int
    out_w: int

    def to_dict(self) -> dict[str, int | bool | str]:
        """Return the transform as a flat dict (for metadata CSVs)."""
        return asdict(self)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected foreground component of a ``uint8`` image, holes filled."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros(binary.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def border_strip_mask(u8: np.ndarray, scan_frac: float, min_frac: float, level: int) -> np.ndarray:
    """Find saturated scanner strips along the image borders.

    Film scanners leave bright, often slightly tilted strips along the edges. They can touch
    the chest wall, so they survive the largest-component step. A strip is a connected
    component of near-saturated pixels (``>= level``) inside the outer ``scan_frac`` band of
    one side (strips don't always start at pixel 0; cutting at the band edge separates a strip
    from adjacent bright tissue) that spans at least ``min_frac`` of the image height
    (left/right bands) or width (top/bottom bands). Tissue never has that shape.

    Returns:
        Boolean mask of strip pixels, same shape as ``u8``.
    """
    h, w = u8.shape
    sx, sy = max(1, round(w * scan_frac)), max(1, round(h * scan_frac))
    found = np.zeros((h, w), dtype=bool)
    bright = u8 >= level
    # (band region, stat measuring the strip's length, length it must span)
    bands = [
        ((slice(None), slice(0, sx)), cv2.CC_STAT_HEIGHT, h),
        ((slice(None), slice(w - sx, w)), cv2.CC_STAT_HEIGHT, h),
        ((slice(0, sy), slice(None)), cv2.CC_STAT_WIDTH, w),
        ((slice(h - sy, h), slice(None)), cv2.CC_STAT_WIDTH, w),
    ]
    for region, stat, length in bands:
        band = bright[region].astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        is_strip = stats[:, stat] >= min_frac * length
        is_strip[0] = False  # background label
        found[region] |= is_strip[labels]
    return found


def segment_breast(img: np.ndarray, params: CropParams) -> tuple[np.ndarray, np.ndarray]:
    """Segment the breast at the working resolution.

    Args:
        img: Original image (any numeric dtype, higher = denser).
        params: Segmentation parameters.

    Returns:
        ``(breast, strips)``: boolean masks of shape
        ``resize_long_side(img, params.work_long_side).shape`` for the breast and for
        saturated border strips (see :func:`border_strip_mask`).
    """
    small = resize_long_side(img.astype(np.float32), params.work_long_side, cv2.INTER_AREA)
    u8 = (normalize_intensity(small, 0.0, params.high_pct) * 255).astype(np.uint8)
    strips = border_strip_mask(
        u8, params.strip_scan_frac, params.strip_min_frac, params.strip_level
    )
    u8[strips] = 0
    h, w = u8.shape
    by, bx = round(h * params.border_trim_frac), round(w * params.border_trim_frac)
    if by or bx:
        u8[:by], u8[h - by :], u8[:, :bx], u8[:, w - bx :] = 0, 0, 0, 0
    if params.blur_ksize > 1:
        u8 = cv2.GaussianBlur(u8, (params.blur_ksize, params.blur_ksize), 0)
    _, binary = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph_ksize,) * 2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _largest_component(binary) & ~strips, strips


def bounding_box(mask: np.ndarray, margin_frac: float) -> tuple[int, int, int, int]:
    """Bounding box ``(y0, y1, x0, x1)`` (end-exclusive) of a mask, padded and clipped."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = round((y1 - y0) * margin_frac), round((x1 - x0) * margin_frac)
    h, w = mask.shape
    return max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)


def chest_wall_side(mask: np.ndarray) -> Side:
    """Return which side of the image the chest wall is on.

    The breast is widest (tallest, in image rows) at the chest wall and tapers to the nipple,
    so the side whose outer quarter holds more breast pixels is the chest-wall side.
    """
    cols = mask.sum(axis=0)
    nz = np.nonzero(cols)[0]
    if nz.size == 0:
        return "left"
    cols = cols[nz[0] : nz[-1] + 1]
    q = max(1, cols.size // 4)
    return "left" if cols[:q].sum() >= cols[-q:].sum() else "right"


def resize_long_side(img: np.ndarray, long_side: int, interpolation: int) -> np.ndarray:
    """Resize so the longer side equals ``long_side``, keeping the aspect ratio."""
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=interpolation)


def compute_transform(
    img: np.ndarray, params: CropParams, long_side: int, canonical_side: Side
) -> tuple[Transform, np.ndarray, np.ndarray]:
    """Compute the crop/flip/resize transform for a mammogram.

    Args:
        img: Original image.
        params: Segmentation parameters.
        long_side: Long side of the output image in pixels.
        canonical_side: Side the chest wall should end up on after flipping.

    Returns:
        The transform, the breast mask and the border-strip mask (both at working resolution).
    """
    mask, strips = segment_breast(img, params)
    h, w = img.shape
    sy, sx = h / mask.shape[0], w / mask.shape[1]
    by0, by1, bx0, bx1 = bounding_box(mask, params.margin_frac)
    y0, y1 = int(by0 * sy), min(h, round(by1 * sy))
    x0, x1 = int(bx0 * sx), min(w, round(bx1 * sx))
    side = chest_wall_side(mask)
    scale = long_side / max(y1 - y0, x1 - x0)
    transform = Transform(
        orig_h=h,
        orig_w=w,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        flipped=side != canonical_side,
        chest_wall_side=side,
        out_h=max(1, round((y1 - y0) * scale)),
        out_w=max(1, round((x1 - x0) * scale)),
    )
    return transform, mask, strips


def apply_transform(
    img: np.ndarray,
    t: Transform,
    interpolation: int,
    background_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Crop, optionally zero the background, flip and resize an image or mask.

    Args:
        img: Original-resolution image or mask. A mask whose shape differs from the image
            the transform was computed on is first resized to ``(orig_h, orig_w)``.
        t: Transform from :func:`compute_transform`.
        interpolation: OpenCV interpolation (``INTER_AREA`` for images, ``INTER_NEAREST``
            for masks).
        background_mask: Optional boolean breast mask (any resolution); pixels outside it are
            set to 0.

    Returns:
        Array of shape ``(t.out_h, t.out_w)`` with the input dtype.
    """
    if img.shape != (t.orig_h, t.orig_w):
        img = cv2.resize(img, (t.orig_w, t.orig_h), interpolation=cv2.INTER_NEAREST)
    out = img[t.y0 : t.y1, t.x0 : t.x1].copy()
    if background_mask is not None:
        full = cv2.resize(
            background_mask.astype(np.uint8), (t.orig_w, t.orig_h), interpolation=cv2.INTER_NEAREST
        )
        out[full[t.y0 : t.y1, t.x0 : t.x1] == 0] = 0
    if t.flipped:
        out = out[:, ::-1]
    return cv2.resize(np.ascontiguousarray(out), (t.out_w, t.out_h), interpolation=interpolation)


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a boolean mask by ``px`` pixels with an elliptical kernel."""
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def preprocess_mammogram(
    img: np.ndarray, params: CropParams, long_side: int, canonical_side: Side
) -> tuple[np.ndarray, Transform, np.ndarray]:
    """Run the full preprocessing on one mammogram.

    Returns:
        The preprocessed image (input dtype), its transform, and the breast mask that was
        used for background zeroing (dilated, strips removed), at working resolution.
    """
    t, mask, strips = compute_transform(img, params, long_side, canonical_side)
    keep = dilate_mask(mask, params.background_dilate_px) & ~strips
    out = apply_transform(img, t, cv2.INTER_AREA, keep if params.zero_background else None)
    return out, t, keep
