# Inferring turbulence screens from only rho(0) and rho(Z)

These are **two independent inversions** of the same held-out trajectory. The
optimization sees only the saved initial intensity and final intensity; saved
screens are loaded solely to score recovery and to run post-fit forward-model
diagnostics. Neither script uses intermediate intensity images as observations.

From the `TFNO` repository on a GPU compute node:

```bash
python invert_screens_fno.py --max-its 1000
python invert_screens_split_step.py --max-its 1000
```

Both commands default to the first **test** path listed in
`checkpoints/turpy_fno_4km_ic_split/split_manifest.json` and read
`data/turpy_chunks_4km/chunk_*.pt`. To compare another path, pass the same
`--path-id ID` to both commands. `--device cuda`, `--learning-rate`,
`--regularization {none,l2,smooth}`, `--alpha`, and `--output-dir` are available
on each script. For a short plumbing test, use `--max-its 3`.

## Whole held-out test set

On a GPU compute node, evaluate every **test path** once at the final plane:

```bash
python invert_screens_fno.py --all-test-paths --max-its 1000
python invert_screens_split_step.py --all-test-paths --max-its 1000
```

For a quick check before the full run, add `--max-paths 2 --max-its 3
--example-plots 0 --output-dir SOME_SEPARATE_SMOKE_DIRECTORY`. Keep smoke and
full results in different directories if their optimizer settings differ.
`--resume` continues an interrupted run with the **same** optimizer/settings;
you may start with `--max-paths 2` and then omit it when resuming to finish
the test set. A fresh run will not overwrite an existing batch directory;
use `--resume` or a new `--output-dir`. Each script reads one chunk at a time and loads its model or
propagator once, not once per path. Full split-step inversion can still take
many GPU-hours; use a short pilot to estimate runtime.

The two batch runs write to separate directories, `fno_screen_test_set/` and
`split_step_screen_test_set/`. Each contains one `path_metrics.jsonl` row per
held-out path, `test_set_summary.json` with mean/median/quantiles for final
intensity and screen recovery, `test_set_summary.png`, and `run_config.json`.
Only the first three test paths get detailed files under `examples/` by
default; set `--example-plots 0` for none. Per-screen aggregate correlations
exclude the unidentifiable final screen. The true screens are used only after
optimization to compute evaluation metrics, including the oracle forward fit.

The FNO script freezes the checkpoint and optimizes its 20 normalized delta-n
input slots. The split-step script instead optimizes phase screens in radians
using TurPy's differentiable coherent propagator, with
`phase = (2*pi/wavelength) * dz * delta_n`. Both enforce zero spatial mean per
screen, removing an unobservable constant-phase offset.

The physical script requires a **flat initial phase**, because rho(0) alone
does not specify a vortex or other non-flat wavefront. It defaults to the
generator's zero padding with factor 2; if the chunk was generated differently,
pass `--no-zero-padding` or the matching `--padding-factor`. Older chunks do
not record those two settings. Check
`oracle_split_step_image_metrics_evaluation_only.relative_l2` in its
`summary.json`: it should be close to zero if the forward model matches the
generator. A large value invalidates the physical comparison.

Each script writes to a **different directory** under the checkpoint folder:

- `fno_screen_inversion_path_ID/`
- `split_step_screen_inversion_path_ID/`

Each directory contains `final_intensity.png`, `phase_screens.png`,
`optimization.png`, `history.json`, `summary.json`, and `fields.pt`. The screen
plots show a representative subset; `fields.pt` contains all 20 inferred and
true screens. `summary.json` reports final-image metrics, per-screen recovery
metrics, and a random-probe gradient sensitivity for each screen. True-screen
forward predictions and screen scores are explicitly **evaluation only**.
Compare the two methods using the final-image relative L2/SSIM/PSNR metrics;
their plotted training losses use different normalizations.

There is no guarantee of recovering the true screens from one final intensity
image: many screen sequences can produce similar images. In particular, the
**last screen cannot affect rho(Z)** in this generator, because it is applied
immediately before the final intensity measurement. It is plotted and scored
only to expose this limitation, not to claim recovery. The split-step script
should report near-zero last-screen sensitivity; if the FNO reports a nonzero
value, that is a learned surrogate artifact rather than physical information.
