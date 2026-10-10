import importlib

import pytest

SUBPACKAGES = [
    "data",
    "models",
    "training",
    "evaluation",
    "explain",
    "uncertainty",
    "serving",
    "utils",
]


def test_version():
    import breast_cancer

    assert breast_cancer.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(f"breast_cancer.{name}")
