"""Resumable download of the CBIS-DDSM collection from TCIA via tcia_utils.

Every series is downloaded into a staging directory (``dicom/.partial/``) and only moved to
``dicom/<SeriesInstanceUID>/`` once its file count matches NBIA's ``ImageCount``. A series is
therefore either complete on disk or absent, so the script can be stopped (Ctrl+C / kill) and
restarted at any time; completed series are skipped.

Usage:
    python scripts/download_cbis_ddsm.py [--out data/raw/cbis_ddsm] [--min-free-gb 15]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from tcia_utils import nbia

COLLECTION = "CBIS-DDSM"
BYTES_PER_GB = 1024**3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw/cbis_ddsm"))
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=15.0,
        help="Stop cleanly when free disk space on the target drive drops below this.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Attempts per series.")
    parser.add_argument("--batch-size", type=int, default=20, help="Series per batch.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent downloads.")
    parser.add_argument(
        "--yes", action="store_true", help="Don't pause after printing the size estimate."
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    """Log to both ``log_path`` (appending) and stdout."""
    logger = logging.getLogger("download_cbis_ddsm")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def fetch_series_list(manifest_path: Path) -> list[dict[str, Any]]:
    """Return the collection's series metadata, cached to ``manifest_path`` as JSON."""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    series = nbia.getSeries(collection=COLLECTION)
    if not series:
        raise RuntimeError(f"NBIA returned no series for collection {COLLECTION!r}.")
    manifest_path.write_text(json.dumps(series, indent=1), encoding="utf-8")
    return series


def count_dicoms(series_dir: Path) -> int:
    """Count ``.dcm`` files anywhere under ``series_dir``."""
    return sum(1 for _ in series_dir.rglob("*.dcm"))


def is_complete(series_dir: Path, expected: int) -> bool:
    """Whether ``series_dir`` exists and holds at least ``expected`` DICOM files."""
    return series_dir.is_dir() and count_dicoms(series_dir) >= expected


def dir_size_bytes(path: Path) -> int:
    """Total size of all files under ``path``."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download_batch(
    batch: list[dict[str, Any]], dicom_dir: Path, staging_dir: Path, workers: int
) -> list[dict[str, Any]]:
    """Download a batch of series concurrently; move verified ones into place.

    Returns the series from ``batch`` that are not complete after this attempt.
    """
    for s in batch:  # clear leftovers from an interrupted run, else tcia_utils skips them
        uid = s["SeriesInstanceUID"]
        shutil.rmtree(staging_dir / uid, ignore_errors=True)
        (staging_dir / f"{uid}.zip").unlink(missing_ok=True)
    nbia.downloadSeries(
        [s["SeriesInstanceUID"] for s in batch],
        input_type="list",
        path=str(staging_dir),
        max_workers=workers,
    )
    incomplete = []
    for s in batch:
        uid = s["SeriesInstanceUID"]
        if is_complete(staging_dir / uid, int(s.get("ImageCount", 1))):
            (staging_dir / uid).rename(dicom_dir / uid)
        else:
            incomplete.append(s)
    return incomplete


def main() -> int:
    """Download every CBIS-DDSM series that is not already complete on disk."""
    args = parse_args()
    out: Path = args.out
    dicom_dir = out / "dicom"
    staging_dir = dicom_dir / ".partial"
    staging_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "download.log")

    series = fetch_series_list(out / "series_manifest.json")
    total = len(series)
    total_bytes = sum(int(s.get("FileSize", 0)) for s in series)
    log.info(
        "%s: %d series, %d images, estimated %.1f GB",
        COLLECTION,
        total,
        sum(int(s.get("ImageCount", 0)) for s in series),
        total_bytes / BYTES_PER_GB,
    )

    todo = [
        s
        for s in series
        if not is_complete(dicom_dir / s["SeriesInstanceUID"], int(s.get("ImageCount", 1)))
    ]
    done = total - len(todo)
    done_bytes = total_bytes - sum(int(s.get("FileSize", 0)) for s in todo)
    remaining_gb = (total_bytes - done_bytes) / BYTES_PER_GB
    free_gb = shutil.disk_usage(out).free / BYTES_PER_GB
    log.info(
        "Already complete: %d/%d (%.1f GB). Remaining: %.1f GB. Free disk: %.1f GB.",
        done,
        total,
        done_bytes / BYTES_PER_GB,
        remaining_gb,
        free_gb,
    )
    if remaining_gb > free_gb - args.min_free_gb:
        log.warning(
            "Remaining download (%.1f GB) does not fit in free space (%.1f GB, keeping %.0f GB "
            "reserve). The script will stop cleanly when the reserve is reached.",
            remaining_gb,
            free_gb,
            args.min_free_gb,
        )
    if not args.yes and sys.stdin.isatty():
        input("Press Enter to start (Ctrl+C to abort)... ")

    failed: list[dict[str, Any]] = []
    for i in range(0, len(todo), args.batch_size):
        if shutil.disk_usage(out).free / BYTES_PER_GB < args.min_free_gb:
            log.error(
                "Free disk below %.0f GB reserve; stopping. Rerun to resume.", args.min_free_gb
            )
            return 2
        batch = todo[i : i + args.batch_size]
        start = time.monotonic()
        pending = batch
        for attempt in range(1, args.retries + 1):
            pending = download_batch(pending, dicom_dir, staging_dir, args.workers)
            if not pending:
                break
            log.warning("%d series incomplete (attempt %d/%d)", len(pending), attempt, args.retries)
        failed.extend(pending)
        finished = [s for s in batch if s not in pending]
        done += len(finished)
        done_bytes += sum(dir_size_bytes(dicom_dir / s["SeriesInstanceUID"]) for s in finished)
        log.info(
            "[%d/%d series] %.2f / %.1f GB (%.1f%%) | batch of %d in %.0fs | failed so far: %d",
            done,
            total,
            done_bytes / BYTES_PER_GB,
            total_bytes / BYTES_PER_GB,
            100 * done_bytes / max(total_bytes, 1),
            len(batch),
            time.monotonic() - start,
            len(failed),
        )

    if failed:
        uids = ", ".join(s["SeriesInstanceUID"] for s in failed[:10])
        log.error("%d series failed; rerun to retry them. First few: %s", len(failed), uids)
        return 1
    log.info("All %d series downloaded.", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
