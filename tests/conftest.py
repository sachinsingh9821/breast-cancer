"""Shared fixtures: synthetic mammograms and DICOM files."""

import json
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


MASS_COLS = (
    "patient_id,breast_density,left or right breast,image view,abnormality id,abnormality type,"
    "mass shape,mass margins,assessment,pathology,subtlety,image file path,"
    "cropped image file path,ROI mask file path"
)
CALC_COLS = (
    "patient_id,breast density,left or right breast,image view,abnormality id,abnormality type,"
    "calc type,calc distribution,assessment,pathology,subtlety,image file path,"
    "cropped image file path,ROI mask file path"
)


def build_fake_cbis(tmp_path: Path) -> Path:
    """Write a miniature CBIS-DDSM download reproducing the real dataset's quirks.

    * P_00001 LEFT CC appears under a Mass and a Calc subject with identical pixels (duplicate).
      Its mass ROI series holds the mask under a wrong "cropped images" description and the
      CSV's 00000N indices are swapped; its calc crop and mask are in separate series.
    * P_00002 RIGHT MLO (calc test): ROI path has a trailing newline; chest wall on the right.
    * P_00003 is in both the official train (LEFT CC) and test (LEFT MLO) sets.
    * P_00004 LEFT CC / MLO and P_00005 RIGHT CC: ordinary train cases.
    """
    root = tmp_path / "cbis"
    dicom, csv_dir = root / "dicom", root / "csv"
    csv_dir.mkdir(parents=True)
    manifest, rows = [], {k: [] for k in ("mass_train", "mass_test", "calc_train", "calc_test")}
    counter = iter(range(1, 10_000))

    def uid() -> str:
        return f"1.2.3.{next(counter)}"

    def series(subject: str, description: str, files: list[tuple[np.ndarray, str]]) -> str:
        s = uid()
        for i, (arr, desc) in enumerate(files, start=1):
            write_dicom(dicom / s / f"1-{i}.dcm", arr, series_description=desc, patient_id=subject)
        manifest.append(
            {
                "SeriesInstanceUID": s,
                "StudyInstanceUID": uid(),
                "PatientID": subject,
                "SeriesDescription": description,
                "ImageCount": len(files),
            }
        )
        return s

    def case(
        table,
        kind,
        split,
        pid,
        side,
        view,
        abn_id,
        pathology,
        img,
        chest="left",
        separate_roi=False,
        swap_desc=False,
        trailing_newline=False,
        full_uid=None,
    ):
        subj = (
            f"{kind.capitalize()}-{'Training' if split == 'train' else 'Test'}_{pid}_{side}_{view}"
        )
        full_uid = full_uid or series(
            subj, "full mammogram images", [(img, "full mammogram images")]
        )
        mask = lesion_mask(chest).astype(np.uint8)
        crop = img[400:600, 100:300].copy()
        roi_subj = f"{subj}_{abn_id}"
        if separate_roi:
            crop_uid = series(roi_subj, "cropped images", [(crop, "cropped images")])
            mask_uid = series(roi_subj, "ROI mask images", [(mask, "ROI mask images")])
            crop_file, mask_file = "000000.dcm", "000000.dcm"
        else:
            mask_desc = "cropped images" if swap_desc else "ROI mask images"
            crop_uid = mask_uid = series(
                roi_subj, "ROI mask images", [(mask, mask_desc), (crop, "cropped images")]
            )
            crop_file, mask_file = "000000.dcm", "000001.dcm"  # CSV index wrong on purpose
        study = "9.9.9"
        mask_path = f"{roi_subj}/{study}/{mask_uid}/{mask_file}" + (
            "\n" if trailing_newline else ""
        )
        extra = "OVAL,CIRCUMSCRIBED" if kind == "mass" else "PLEOMORPHIC,CLUSTERED"
        rows[table].append(
            f"{pid},3,{side},{view},{abn_id},{kind},{extra},4,{pathology},3,"
            f"{subj}/{study}/{full_uid}/000000.dcm,{roi_subj}/{study}/{crop_uid}/{crop_file},"
            f'"{mask_path}"'
        )
        return full_uid

    img = make_mammogram()
    case(
        "mass_train", "mass", "train", "P_00001", "LEFT", "CC", 1, "MALIGNANT", img, swap_desc=True
    )
    case(
        "calc_train",
        "calc",
        "train",
        "P_00001",
        "LEFT",
        "CC",
        1,
        "BENIGN",
        img.copy(),
        separate_roi=True,
    )
    case(
        "calc_test",
        "calc",
        "test",
        "P_00002",
        "RIGHT",
        "MLO",
        1,
        "BENIGN_WITHOUT_CALLBACK",
        make_mammogram("right"),
        chest="right",
        trailing_newline=True,
    )
    case("mass_train", "mass", "train", "P_00003", "LEFT", "CC", 1, "BENIGN", make_mammogram())
    case("mass_test", "mass", "test", "P_00003", "LEFT", "MLO", 1, "MALIGNANT", make_mammogram())
    for view in ("CC", "MLO"):
        case(
            "mass_train", "mass", "train", "P_00004", "LEFT", view, 1, "MALIGNANT", make_mammogram()
        )
    case("calc_train", "calc", "train", "P_00005", "RIGHT", "CC", 1, "BENIGN", make_mammogram())

    names = {
        "mass_train": "mass_case_description_train_set.csv",
        "mass_test": "mass_case_description_test_set.csv",
        "calc_train": "calc_case_description_train_set.csv",
        "calc_test": "calc_case_description_test_set.csv",
    }
    for table, name in names.items():
        header = MASS_COLS if table.startswith("mass") else CALC_COLS
        (csv_dir / name).write_text("\n".join([header, *rows[table]]) + "\n")
    (root / "series_manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def fake_cbis(tmp_path):
    return build_fake_cbis(tmp_path)


@pytest.fixture(scope="module")
def fake_cbis_module(tmp_path_factory):
    return build_fake_cbis(tmp_path_factory.mktemp("fake_cbis"))
