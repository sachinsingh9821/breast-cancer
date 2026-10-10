import numpy as np
import pytest

from breast_cancer.data.dicom_io import normalize_intensity, read_dicom_image, read_header
from conftest import write_dicom


def test_read_monochrome2_keeps_values(tmp_path):
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000
    out = read_dicom_image(write_dicom(tmp_path / "a.dcm", arr))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, arr)


def test_read_monochrome1_is_inverted(tmp_path):
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000
    out = read_dicom_image(write_dicom(tmp_path / "a.dcm", arr, photometric="MONOCHROME1"))
    np.testing.assert_array_equal(out, arr.max() - arr)


def test_read_8bit_mask(tmp_path):
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[2:4, 2:4] = 255
    out = read_dicom_image(write_dicom(tmp_path / "m.dcm", arr, "MONOCHROME2", "ROI mask images"))
    assert out.max() == 255 and (out > 0).sum() == 4


def test_read_header_skips_pixels(tmp_path):
    path = write_dicom(tmp_path / "a.dcm", np.zeros((4, 4), np.uint16))
    ds = read_header(path)
    assert ds.SeriesDescription == "full mammogram images"
    assert "PixelData" not in ds


def test_normalize_intensity_range():
    img = np.linspace(0, 1000, 10001, dtype=np.float32)
    out = normalize_intensity(img, 1.0, 99.0)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[len(out) // 2] == pytest.approx(0.5, abs=1e-3)


def test_normalize_intensity_uses_mask():
    img = np.zeros((10, 10), dtype=np.float32)
    img[:5] = np.linspace(100, 200, 50).reshape(5, 10)
    mask = img > 0
    out = normalize_intensity(img, 0.0, 100.0, mask=mask)
    assert out[mask].max() == 1.0
    assert out[0, 0] == 0.0  # value 100 is the masked minimum
