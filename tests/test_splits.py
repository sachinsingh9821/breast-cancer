from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breast_cancer.data.splits import check_no_patient_leakage, make_splits

REAL_SPLITS = Path(__file__).resolve().parents[1] / "data/processed/cbis_ddsm/splits.csv"


def synthetic_images(n_patients: int = 400, n_overlap: int = 10, seed: int = 0) -> pd.DataFrame:
    """Image table shaped like CBIS: ~25% official test, some patients in both sets."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"P_{p:05d}"
        test = p % 4 == 0
        kind = rng.choice(["mass", "calc", "calc+mass"], p=[0.5, 0.45, 0.05])
        malignant = int(rng.random() < 0.45)
        views = [["CC", "MLO"], ["MLO"], ["CC"]][rng.choice(3, p=[0.8, 0.15, 0.05])]
        for view in views:
            rows.append((pid, view, "test" if test else "train", kind, malignant))
        if 0 < p <= n_overlap * 4 and p % 4 == 1:  # train patient that also has a test image
            rows.append((pid, "XCC", "test", kind, malignant))
    df = pd.DataFrame(rows, columns=["patient_id", "view", "official_split", "kind", "label"])
    df["image_id"] = df["patient_id"] + "_LEFT_" + df["view"]
    df["abnormality_types"] = df["kind"]
    df["in_both_official_splits"] = False
    return df


@pytest.fixture
def splits():
    images = synthetic_images()
    return images, *make_splits(images, val_frac=0.15, seed=42)


def test_no_patient_in_more_than_one_split(splits):
    _, s, _ = splits
    used = s[s.split.isin(["train", "val", "test"])]
    assert (used.groupby("patient_id").split.nunique() == 1).all()


def test_official_test_set_is_kept(splits):
    images, s, _ = splits
    merged = images.merge(s, on=["image_id", "patient_id"])
    assert (merged.loc[merged.official_split == "test", "split"] == "test").all()
    assert (merged.loc[merged.split == "test", "official_split"] == "test").all()


def test_overlap_patients_dropped_from_train(splits):
    images, s, dropped = splits
    assert len(dropped) == 10
    overlap = set(dropped.patient_id)
    assert not s[s.patient_id.isin(overlap) & s.split.isin(["train", "val"])].any().any()
    assert (s[s.split == "dropped"].patient_id.isin(overlap)).all()
    assert dropped.n_train_images_dropped.sum() == (s.split == "dropped").sum()
    assert (dropped.n_test_images_kept >= 1).all()


def test_val_fraction_and_stratification(splits):
    _, s, _ = splits
    pool = s[s.split.isin(["train", "val"])].drop_duplicates("patient_id")
    frac = (pool.split == "val").mean()
    assert frac == pytest.approx(0.15, abs=0.02)
    by_stratum = pool.groupby("stratum").split.apply(lambda x: (x == "val").mean())
    big = pool.stratum.value_counts()
    for stratum in big[big >= 40].index:
        assert by_stratum[stratum] == pytest.approx(0.15, abs=0.05)


def test_splits_are_deterministic():
    images = synthetic_images()
    a, _ = make_splits(images, 0.15, seed=1)
    b, _ = make_splits(images, 0.15, seed=1)
    c, _ = make_splits(images, 0.15, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.split.equals(c.split)


def test_leakage_check_raises():
    bad = pd.DataFrame(
        {"image_id": ["a", "b"], "patient_id": ["P1", "P1"], "split": ["train", "val"]}
    )
    with pytest.raises(ValueError, match="more than one split"):
        check_no_patient_leakage(bad)


def test_fake_cbis_end_to_end(fake_cbis, tmp_path):
    from breast_cancer.data.cbis_index import build_index

    images, _ = build_index(
        fake_cbis / "csv", fake_cbis / "series_manifest.json", fake_cbis / "dicom", tmp_path
    )
    s, dropped = make_splits(images, val_frac=0.34, seed=0)
    assert dropped.patient_id.tolist() == ["P_00003"]
    by_id = s.set_index("image_id").split
    assert by_id["P_00003_LEFT_CC"] == "dropped"
    assert by_id["P_00003_LEFT_MLO"] == "test"
    assert by_id["P_00002_RIGHT_MLO"] == "test"


@pytest.mark.skipif(not REAL_SPLITS.exists(), reason="real CBIS-DDSM splits not built yet")
def test_real_cbis_splits_have_no_patient_leakage():
    s = pd.read_csv(REAL_SPLITS)
    check_no_patient_leakage(s)
    used = s[s.split.isin(["train", "val", "test"])]
    assert (used.groupby("patient_id").split.nunique() == 1).all()
