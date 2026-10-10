import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from breast_cancer.data.pipeline import mask_bbox, read_png

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def preprocessed(fake_cbis_module, tmp_path_factory):
    """Run scripts/preprocess.py (all stages) once on the fake CBIS download."""
    fake_cbis, tmp_path = fake_cbis_module, tmp_path_factory.mktemp("pipeline")
    processed = tmp_path / "processed"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preprocess.py"),
        f"data.raw_dir={fake_cbis.as_posix()}",
        f"data.interim_dir={(tmp_path / 'interim').as_posix()}",
        f"data.processed_dir={processed.as_posix()}",
        f"paths.reports={(tmp_path / 'reports').as_posix()}",
        "data.master_long_side=512",
        "data.derived_long_sides=[256,128]",
        "data.long_side=256",
        "data.crop.work_long_side=512",
        "data.crop.morph_ksize=9",
        "data.split.val_frac=0.34",
        "data.num_workers=2",
        f"hydra.run.dir={(tmp_path / 'hydra').as_posix()}",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    return processed, tmp_path


def test_all_outputs_written(preprocessed):
    processed, tmp = preprocessed
    for rel in [
        "splits.csv",
        "dropped_overlap_patients.csv",
        "preprocess_run.json",
        "master_512/metadata.csv",
        "master_512/lesions.csv",
        "ls256/metadata.csv",
        "ls128/lesions.csv",
    ]:
        assert (processed / rel).exists(), rel
    assert (tmp / "reports" / "cbis_ddsm_dropped_overlap_patients.csv").exists()
    run = json.loads((processed / "preprocess_run.json").read_text())
    assert run["git_commit"] and run["config"]["master_long_side"] == 512


def test_master_images_and_metadata(preprocessed):
    processed, _ = preprocessed
    master = processed / "master_512"
    meta = pd.read_csv(master / "metadata.csv")
    assert len(meta) == 7
    expected = {"patient_id", "laterality", "view", "label", "pathology", "breast_density"}
    assert expected <= set(meta.columns)
    assert {"full_file", "flipped", "chest_wall_side", "y0", "x1"} <= set(meta.columns)
    for row in meta.itertuples():
        img = read_png(master / row.image_path)
        assert img.dtype == np.uint16
        assert max(img.shape) == 512 and img.shape == (row.out_h, row.out_w)
    m = meta.set_index("image_id")
    assert m.loc["P_00002_RIGHT_MLO", "flipped"]  # right chest wall moved to the left
    assert not m.loc["P_00004_LEFT_CC", "flipped"]


def test_masks_align_with_lesions(preprocessed):
    processed, _ = preprocessed
    master = processed / "master_512"
    lesions = pd.read_csv(master / "lesions.csv")
    assert len(lesions) == 8
    assert (lesions["mask_status"] == "ok").all()
    for les in lesions.itertuples():
        img = read_png(master / "images" / f"{les.image_id}.png")
        mask = read_png(master / les.mask_path)
        assert mask.shape == img.shape and mask.dtype == np.uint8
        assert img[mask > 0].mean() > 40000  # mask sits on the bright synthetic lesion
        assert mask_bbox(mask)["mask_area"] == les.mask_area > 0


def test_derived_copies(preprocessed):
    processed, _ = preprocessed
    for ls in (256, 128):
        d = processed / f"ls{ls}"
        meta = pd.read_csv(d / "metadata.csv")
        lesions = pd.read_csv(d / "lesions.csv")
        assert len(meta) == 7 and len(lesions) == 8
        for row in meta.itertuples():
            img = read_png(d / row.image_path)
            assert max(img.shape) == ls and img.shape == (row.out_h, row.out_w)
        for les in lesions.itertuples():
            mask = read_png(d / les.mask_path)
            assert les.mask_area > 0 and set(np.unique(mask)) <= {0, 255}


def test_splits_file(preprocessed):
    processed, _ = preprocessed
    s = pd.read_csv(processed / "splits.csv").set_index("image_id")["split"]
    assert s["P_00003_LEFT_CC"] == "dropped" and s["P_00003_LEFT_MLO"] == "test"


def test_master_is_resumable(preprocessed, fake_cbis_module):
    from breast_cancer.data.pipeline import build_master
    from breast_cancer.data.preprocessing import CropParams

    processed, tmp = preprocessed
    master = processed / "master_512"
    (master / "records" / "P_00004_LEFT_CC.json").unlink()
    before = {p.name: p.stat().st_mtime_ns for p in (master / "images").glob("*.png")}
    images = pd.read_csv(tmp / "interim" / "images.csv")
    abn = pd.read_csv(tmp / "interim" / "abnormalities.csv")
    params = CropParams(512, 99.0, 0.02, 0.06, 0.5, 250, 5, 9, 0.01, True, 5)
    meta, _ = build_master(images, abn, fake_cbis_module / "dicom", master, params, 512, "left", 1)
    after = {p.name: p.stat().st_mtime_ns for p in (master / "images").glob("*.png")}
    changed = [k for k in before if before[k] != after[k]]
    assert changed == ["P_00004_LEFT_CC.png"]
    assert len(meta) == 7
