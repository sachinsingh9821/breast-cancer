"""Shared fixtures: synthetic mammograms and DICOM files."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

H, W = 1200, 700
LESION_CENTER = (500, 180)  # (row, col) for a breast with the chest wall on the left


def make_mammogram(
    chest_wall: str = "left",
    strip: bool = True,
    label: bool = True,
    lesion: bool = True,
) -> np.ndarray:
    """Synthetic uint16 mammogram: half-ellipse breast, optional scanner strip / label / lesion.

    The breast is attached to the ``chest_wall`` side. The strip is a tilted saturated band on
    the chest-wall edge (where it touches the breast), the label a bright block in a corner
    of the opposite side, and the lesion a bright disk at ``LESION_CENTER`` (mirrored for a
    right chest wall).
    """
    img = np.zeros((H, W), dtype=np.uint16)
    breast = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(breast, (0, H // 2), (450, 480), 0, -90, 90, 1, thickness=-1)
    rng = np.random.default_rng(0)
    img[breast == 1] = rng.integers(18000, 30000, size=int(breast.sum()), dtype=np.uint16)
    if lesion:
        cv2.circle(img, (LESION_CENTER[1], LESION_CENTER[0]), 25, 45000, thickness=-1)
    if label:
        img[40:120, W - 160 : W - 40] = 60000
    if strip:
        for r in range(H):  # 12 px wide, drifting 8 px over the height
            c0 = 2 + (8 * r) // H
            img[r, c0 : c0 + 12] = 65535
    if chest_wall == "right":
        img = img[:, ::-1].copy()
    return img


def lesion_mask(chest_wall: str = "left") -> np.ndarray:
    """Binary uint8 mask (0/255) of the synthetic lesion."""
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m, (LESION_CENTER[1], LESION_CENTER[0]), 25, 255, thickness=-1)
    return m[:, ::-1].copy() if chest_wall == "right" else m


def write_dicom(
    path: Path,
    arr: np.ndarray,
    photometric: str = "MONOCHROME2",
    series_description: str = "full mammogram images",
    patient_id: str = "Mass-Training_P_00001_LEFT_CC",
    laterality: str = "L",
) -> Path:
    """Write ``arr`` (uint8 or uint16) as a minimal valid DICOM file."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = patient_id
    ds.SeriesDescription = series_description
    ds.Laterality = laterality
    ds.Modality = "MG"
    bits = arr.dtype.itemsize * 8
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    ds.PixelData = np.ascontiguousarray(arr).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def mammogram() -> np.ndarray:
    return make_mammogram()
