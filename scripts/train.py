"""CLI entry point: train (implemented in a later phase)."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the train step with the composed Hydra config."""
    raise NotImplementedError("scripts/train.py is not implemented yet.")


if __name__ == "__main__":
    main()
