import cv2
import numpy as np
import pytest

from breast_cancer.data.preprocessing import (
    CropParams,
    Transform,
    apply_transform,
    border_strip_mask,
    chest_wall_side,
    preprocess_mammogram,
    resize_long_side,
    segment_breast,
)
from conftest import LESION_CENTER, H, W, lesion_mask, make_mammogram

PARAMS = CropParams(
    work_long_side=600,
    high_pct=99.0,
    border_trim_frac=0.02,
    strip_scan_frac=0.06,
    strip_min_frac=0.5,
    strip_level=250,
    blur_ksize=5,
    morph_ksize=9,
    margin_frac=0.01,
    zero_background=True,
    background_dilate_px=3,
)
LONG = 512


def test_resize_long_side_keeps_aspect():
    out = resize_long_side(np.zeros((1000, 400), np.uint16), 500, cv2.INTER_AREA)
    assert out.shape == (500, 200)
    out = resize_long_side(np.zeros((300, 900), np.uint16), 450, cv2.INTER_AREA)
    assert out.shape == (150, 450)


@pytest.mark.parametrize("side", ["left", "right"])
def test_chest_wall_side(side):
    mask, _ = segment_breast(make_mammogram(side), PARAMS)
    assert chest_wall_side(mask) == side


def test_border_strip_detected_but_not_tissue():
    u8 = np.zeros((200, 100), np.uint8)
    u8[:, 1:4] = 255  # full-height strip in the left band
    u8[80:120, 40:60] = 255  # bright tissue blob in the middle
    u8[0:20, 0:5] = 255  # short bright patch in the band (not a strip)
    strips = border_strip_mask(u8, scan_frac=0.06, min_frac=0.5, level=250)
    assert strips[:, 1:4][20:].all()
    assert not strips[80:120, 40:60].any()
    assert not strips[100, 4:].any()


@pytest.mark.parametrize("side", ["left", "right"])
def test_preprocess_flips_to_canonical_and_resizes(side):
    out, t, _ = preprocess_mammogram(make_mammogram(side), PARAMS, LONG, "left")
    assert t.flipped == (side == "right")
    assert t.chest_wall_side == side
    assert max(out.shape) == LONG
    assert out.shape == (t.out_h, t.out_w)
    assert out.dtype == np.uint16
    # chest wall now on the left: the left quarter holds more breast than the right quarter
    q = out.shape[1] // 4
    assert (out[:, :q] > 0).sum() > (out[:, -q:] > 0).sum()


def test_preprocess_removes_strip_and_label():
    img = make_mammogram("left")
    out, t, _ = preprocess_mammogram(img, PARAMS, LONG, "left")
    assert out.max() < 65535  # neither saturated strip nor label (60000) survives
    assert out.max() >= 45000 * 0.9  # lesion is kept
    # the crop starts at the breast, not at the image border
    assert t.y0 > 0 and t.y1 < H and t.x1 < W


def test_preprocess_without_artifacts_crops_tightly():
    img = make_mammogram("left", strip=False, label=False)
    _, t, _ = preprocess_mammogram(img, PARAMS, LONG, "left")
    # ellipse spans rows 120..1080 and cols 0..450 at original resolution
    assert t.y0 == pytest.approx(120, abs=25) and t.y1 == pytest.approx(1080, abs=25)
    assert t.x1 == pytest.approx(450, abs=25)


@pytest.mark.parametrize("side", ["left", "right"])
def test_mask_stays_aligned_with_image(side):
    img = make_mammogram(side, strip=False, label=False)
    out, t, _ = preprocess_mammogram(img, PARAMS, LONG, "left")
    m = apply_transform(lesion_mask(side), t, cv2.INTER_NEAREST)
    assert m.shape == out.shape
    assert set(np.unique(m)) <= {0, 255}
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    assert out[cy, cx] > 40000  # mask centre lands on the bright lesion
    # lesion column expected from the transform: distance from the crop's chest-wall edge
    lesion_x = LESION_CENTER[1] if side == "left" else W - 1 - LESION_CENTER[1]
    dist = lesion_x - t.x0 if not t.flipped else t.x1 - 1 - lesion_x
    assert cx == pytest.approx(dist * t.out_w / (t.x1 - t.x0), abs=3)


def test_apply_transform_resizes_mismatched_mask():
    t = Transform(100, 50, 10, 90, 5, 45, False, "left", 40, 20)
    small = np.full((50, 25), 255, np.uint8)  # half resolution
    assert apply_transform(small, t, cv2.INTER_NEAREST).shape == (40, 20)


def test_transform_to_dict_roundtrip():
    t = Transform(100, 50, 10, 90, 5, 45, True, "right", 40, 20)
    assert Transform(**t.to_dict()) == t
