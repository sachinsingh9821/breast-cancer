import random

import numpy as np
import torch

from breast_cancer.utils.repro import get_git_commit, set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_git_commit_outside_repo(tmp_path):
    assert get_git_commit(tmp_path) == "unknown"
