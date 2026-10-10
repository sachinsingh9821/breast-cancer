"""CLI entry point: export (implemented in a later phase)."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the export step with the composed Hydra config."""
    raise NotImplementedError("scripts/export.py is not implemented yet.")


if __name__ == "__main__":
    main()
