"""Reproducibility helpers: seeding and run metadata (git commit)."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: Seed value applied to every RNG.
        deterministic: If True, also request deterministic cuDNN kernels (slower).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a core dependency
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit(repo_dir: str | Path | None = None) -> str:
    """Return the current git commit hash, suffixed with ``-dirty`` if there are local changes.

    Args:
        repo_dir: Directory inside the repository; defaults to the current working directory.

    Returns:
        The commit hash, or ``"unknown"`` if git or the repository is unavailable.
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit
