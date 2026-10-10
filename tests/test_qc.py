import numpy as np

from breast_cancer.data.qc import QCParams, detect_edge_strip

P = QCParams(work_long_side=1024, strip_band_frac=0.04, strip_level_frac=0.95, strip_min_frac=0.3)
H, W = 800, 400


def breast() -> np.ndarray:
    img = np.zeros((H, W), np.uint16)
    img[50:750, :300] = 30000
    return img


def test_clean_image_not_flagged():
    assert not detect_edge_strip(breast(), P)


def test_thin_vertical_strip_flagged():
    img = breast()
    img[100:700, 2:7] = 65535  # 5 px strip inside the 16 px left band
    assert detect_edge_strip(img, P)


def test_top_edge_bar_flagged():
    img = breast()
    img[1:5, 20:200] = 65535  # horizontal bar along the top edge, half the width
    assert detect_edge_strip(img, P)


def test_pectoral_muscle_not_flagged():
    img = breast()
    for r in range(0, 500):  # bright triangle wider than the band along the chest wall
        img[r, : max(0, 200 - r // 2)] = 65535
    assert not detect_edge_strip(img, P)


def test_short_bright_spot_not_flagged():
    img = breast()
    img[300:340, 2:7] = 65535  # bright, but only 5% of the height
    assert not detect_edge_strip(img, P)


def test_empty_image():
    assert not detect_edge_strip(np.zeros((H, W), np.uint16), P)
