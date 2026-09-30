# TurPy dataset generation

`generate_turpy_datasets.py` creates one-step intensity prediction examples from
TurPy propagation trajectories. Run commands from the project directory, where
the local `turpy` package is available. Python needs PyTorch and NumPy installed.

## Generate a chunk

```bash
python generate_turpy_datasets.py --output turpy_chunks/chunk_000.pt \
  --n-paths 100 --path-start 0
```

Each path has `n_z - 1` propagation steps, so the defaults (`n_z=21` and
`n_paths=100`) produce 2,000 examples. The default configuration uses a 64 × 64
grid, a 4 km propagation distance, and fixed `cn2=1e-15`. Generation defaults
to Gaussian-Bessel initial beams, a flat initial phase, normalized power, and
zero-padded propagation.

To use a different positive turbulence-strength value, pass it in SI units:

```bash
python generate_turpy_datasets.py --output turpy_chunks/cn2_5e-16.pt \
  --n-paths 100 --cn2 5e-16
```

For free-space propagation with no atmospheric turbulence, generate zero phase
screens with:

```bash
python generate_turpy_datasets.py --output turpy_chunks/no_turbulence.pt \
  --n-paths 100 --no-turbulence
```

This keeps the propagation steps but sets every `delta_n` screen to zero. The
saved metadata marks `no_turbulence` as true and records `cn2` and `r0` as
undefined (`None`). `--cn2 0` itself is not accepted. To vary turbulence strength
between intervals instead, use `--random-r0 --r0-min 0.03 --r0-max 0.15`; that
samples the Fried parameter `r0` and still generates turbulence.

To generate more chunks, give each a distinct output filename and non-overlapping
path range. For example:

```bash
python generate_turpy_datasets.py --output turpy_chunks/chunk_001.pt \
  --n-paths 100 --path-start 100
```

Path IDs determine the beam-mode combinations when `--bessel-order-pool` is
enabled, so keep the same settings and seed across chunks. Each chunk records its
configuration and path metadata.

## Merge chunks and make splits

```bash
python generate_turpy_datasets.py --mode merge \
  --chunk-dir turpy_chunks --output turpy_step_dataset.pt
```

Merge checks that chunks have compatible settings and non-overlapping path IDs.
It creates train, validation, and test indices by **path** (80%, 10%, 10% by
default), keeping examples from the same propagation path in one split. Adjust
the fractions with `--train-fraction` and `--val-fraction`.

## Saved data

The `.pt` file is a dictionary containing:

- `X`: inputs shaped `[examples, height, width, channels]`. Channels contain the
  initial intensity, one slot per propagation interval (known `delta_n` screens
  followed by zero-valued future slots), and the available-history fraction.
- `Y`: next-step intensity targets shaped `[examples, height, width, 1]`.
- `path_ids` and `path_metadata`: the source path for each example and its
  generation metadata.
- `splits`: `train_idx`, `val_idx`, and `test_idx` (added when chunks are merged).

Use `--help` to see all generation options, including grid and propagation
settings, random `r0` sampling (`--random-r0`), beam type, and initial phase.
