import pandas as pd
import pytest

from breast_cancer.data.cbis_index import build_index, classify_role, load_case_descriptions


@pytest.fixture
def index(fake_cbis, tmp_path):
    images, abn = build_index(
        fake_cbis / "csv",
        fake_cbis / "series_manifest.json",
        fake_cbis / "dicom",
        tmp_path / "interim",
        workers=2,
    )
    return images, abn, tmp_path / "interim"


def test_load_case_descriptions_normalizes_quirks(fake_cbis):
    cases = load_case_descriptions(fake_cbis / "csv")
    assert len(cases) == 8
    assert "breast_density" in cases and "breast density" not in cases
    assert cases["breast_density"].notna().all()  # filled from both column spellings
    assert not cases["mask_csv_path"].str.contains("\n").any()
    assert set(cases["abnormality_type"]) == {"mass", "calc"}
    assert set(cases["official_split"]) == {"train", "test"}
    assert cases["malignant"].sum() == 4
    assert cases["abnormality_uid"].is_unique


def test_classify_role():
    assert classify_role("full mammogram images") == "full"
    assert classify_role(" ROI mask images ") == "mask"
    assert classify_role("cropped images") == "crop"
    assert classify_role("something else") == "unknown"


def test_every_lesion_gets_full_image_and_mask(index):
    _, abn, _ = index
    assert abn["full_file"].notna().all()
    assert abn["mask_file"].notna().all()
    assert abn["crop_file"].notna().all()
    assert (abn["mask_file"] != abn["crop_file"]).all()


def test_mask_identified_by_size_not_description_or_csv_index(index, fake_cbis):
    _, abn, _ = index
    row = abn[(abn.patient_id == "P_00001") & (abn.abnormality_type == "mass")].iloc[0]
    # the mask is 1-1.dcm although its header says "cropped images" and the CSV says 000001
    assert row["mask_file"].endswith("/1-1.dcm")
    assert row["crop_file"].endswith("/1-2.dcm")
    assert row["mask_resolution"] == "size_overrides_header"


def test_separate_crop_and_mask_series(index):
    _, abn, _ = index
    row = abn[(abn.patient_id == "P_00001") & (abn.abnormality_type == "calc")].iloc[0]
    assert row["mask_file"].split("/")[0] != row["crop_file"].split("/")[0]


def test_duplicate_full_images_are_merged(index):
    images, abn, _ = index
    dup = images[images.patient_id == "P_00001"]
    assert len(dup) == 1
    img = dup.iloc[0]
    assert img["image_id"] == "P_00001_LEFT_CC"
    assert img["abnormality_types"] == "calc+mass"
    assert img["n_abnormalities"] == 2
    assert img["label"] == 1 and img["pathology"] == "MALIGNANT"
    assert len(img["series_uids"].split(";")) == 2
    assert (abn[abn.patient_id == "P_00001"]["image_id"] == "P_00001_LEFT_CC").all()


def test_image_table(index):
    images, _, out = index
    assert len(images) == 7  # 8 lesions, one duplicate image merged
    assert images["image_id"].is_unique
    assert images["available"].all()
    p2 = images.set_index("image_id").loc["P_00002_RIGHT_MLO"]
    assert p2["official_split"] == "test" and p2["label"] == 0
    for name in ("images.csv", "abnormalities.csv", "dicom_headers.csv"):
        assert (out / name).exists()


def test_header_cache_is_reused(index, fake_cbis, tmp_path, monkeypatch):
    _, _, out = index
    import breast_cancer.data.cbis_index as mod

    def boom(*args, **kwargs):
        raise AssertionError("header re-read although cached")

    monkeypatch.setattr(mod, "read_header", boom)
    images, _ = build_index(
        fake_cbis / "csv", fake_cbis / "series_manifest.json", fake_cbis / "dicom", out, workers=1
    )
    assert len(images) == 7  # 8 lesions, one duplicate image merged


def test_partial_download_marks_unavailable(fake_cbis, tmp_path):
    import shutil

    images, _ = build_index(
        fake_cbis / "csv",
        fake_cbis / "series_manifest.json",
        fake_cbis / "dicom",
        tmp_path / "a",
        workers=1,
    )
    victim = images.set_index("image_id").loc["P_00005_RIGHT_CC", "series_uid"]
    shutil.rmtree(fake_cbis / "dicom" / victim)
    images, abn = build_index(
        fake_cbis / "csv",
        fake_cbis / "series_manifest.json",
        fake_cbis / "dicom",
        tmp_path / "b",
        workers=1,
    )
    assert not images.set_index("image_id").loc["P_00005_RIGHT_CC", "available"]
    assert pd.isna(abn.set_index("patient_id").loc["P_00005", "full_file"])
