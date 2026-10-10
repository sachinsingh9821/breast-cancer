from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _compose(overrides=None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_composes():
    cfg = _compose()
    assert cfg.seed == 42
    assert cfg.data.name == "cbis_ddsm"
    assert cfg.train.amp is True


def test_whole_image_resolution_is_high():
    assert _compose().data.long_side >= 1024


@pytest.mark.parametrize("dataset", ["cbis_ddsm", "vindr"])
def test_dataset_configs_compose(dataset):
    assert _compose([f"data={dataset}"]).data.name == dataset


def test_baseline_experiment_overrides_resolution():
    assert _compose(["experiment=baseline"]).data.long_side == 512


def test_cbis_master_and_derived_resolutions():
    data = _compose().data
    assert data.master_long_side == 2048
    assert data.long_side in data.derived_long_sides
    assert _compose(["experiment=baseline"]).data.long_side in data.derived_long_sides
    assert max(data.derived_long_sides) <= data.master_long_side
