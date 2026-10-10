import numpy as np
import pytest

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


def labeled_review(n: int = 300, n_strip: int = 15, n_line: int = 5):
    import pandas as pd

    df = pd.DataFrame({"k": range(n), "image_id": [f"P_{i:05d}_LEFT_CC" for i in range(n)]})
    df["edge_strip"] = [1 if i < n_strip else 0 for i in range(n)]
    df["horizontal_line"] = [1 if n_strip <= i < n_strip + n_line else 0 for i in range(n)]
    return df


def test_verification_sample_is_enriched_blind_and_shuffled():
    from breast_cancer.data.qc import select_verification_sample

    review = labeled_review()
    s = select_verification_sample(review, n=30, max_positive_frac=0.4, seed=0)
    assert len(s) == 30 and s.image_id.is_unique
    assert (s.edge_strip == "").all() and (s.horizontal_line == "").all()  # blind
    labels = review.set_index("image_id").loc[s.image_id]
    n_pos = int((labels.edge_strip | labels.horizontal_line).sum())
    assert n_pos == 12  # 40% of 30, enriched far above the 6.7% base rate
    assert not labels.edge_strip.iloc[:12].all()  # positives are not all at the top


def test_verification_sample_requires_full_labels():
    from breast_cancer.data.qc import select_verification_sample

    review = labeled_review().astype({"edge_strip": float})
    review.loc[3, "edge_strip"] = np.nan
    with pytest.raises(ValueError):
        select_verification_sample(review, 30, 0.4, 0)


def test_score_verification_agreement_and_disagreements():
    from breast_cancer.data.qc import score_verification, select_verification_sample

    first = labeled_review()
    second = select_verification_sample(first, 30, 0.4, 0)
    truth = first.set_index("image_id").loc[second.image_id]
    second["edge_strip"] = truth.edge_strip.astype(str).values
    second["horizontal_line"] = truth.horizontal_line.astype(str).values
    flipped = second.image_id.iloc[0]
    second.loc[0, "edge_strip"] = str(1 - int(second.loc[0, "edge_strip"]))
    r = score_verification(first, second)
    assert r["n_images"] == 30
    assert r["edge_strip"]["agreement"] == pytest.approx(29 / 30, abs=1e-3)
    assert r["edge_strip"]["disagreements"] == [flipped]
    assert r["horizontal_line"]["agreement"] == 1.0
    assert r["horizontal_line"]["cohen_kappa"] == 1.0


def test_score_verification_rejects_unlabeled_rows():
    from breast_cancer.data.qc import score_verification, select_verification_sample

    first = labeled_review()
    second = select_verification_sample(first, 30, 0.4, 0)  # still blank
    with pytest.raises(ValueError, match="unlabeled"):
        score_verification(first, second)
