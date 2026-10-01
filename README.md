# TurPy neural-operator experiments

This repository generates TurPy propagation data and trains and evaluates
Fourier neural operators (FNOs). Run the commands in the guides from the
repository root, including when submitting Slurm jobs.

## Guides

- [Generate turbulent or free-space data](docs/TURPY_DATA_GENERATION_README.md)
- [Train the autoregressive window FNO](docs/WINDOW_FNO_README.md)
- [Invert the initial intensity with an FNO](docs/INVERSION_README.md)
- [Compare screen inversion methods](docs/SCREEN_INVERSION_README.md)
- [Split-step screen inversion and mathematics](docs/SPLIT_STEP_SCREEN_INVERSION_README.md)

## Layout

- `generate_turpy_datasets.py`, `train.py`, and `train_window_fno.py` are the
  data generation and training entry points.
- `slurm/` contains the training job scripts; submit them from the repository
  root, for example `sbatch slurm/train_window_fno.sbatch`.
- `data/` holds generated datasets; see [data/README.md](data/README.md).
- `checkpoints/` holds training outputs, and `tests/` contains repository tests.

Generated `.pt` files, checkpoints, and job logs are excluded from Git.
