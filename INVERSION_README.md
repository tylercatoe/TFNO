# TurPy FNO inversion and rollout diagnostics

`run_turpy_inversion.py` recovers a nonnegative initial intensity `rho(0)` by
optimizing the input to the frozen FNO until its predicted intensity matches
an observed intensity at a selected propagation step. The delta-n screens and
history fraction come from the same saved TurPy example; only `rho(0)` changes.

From `/scratch/tcatoe/Turb_NO_26/TFNO` on a GPU compute node:

```bash
module load ngc/pytorch/23.06
python run_turpy_inversion.py --max-its 500
```

The defaults use `checkpoints/turpy_fno_4km_ic_split/best.pt`, the first **test**
path in that checkpoint directory's `split_manifest.json`, and its final
4 km observation from `turpy_chunks_4km/chunk_*.pt`. The script loads one chunk
at a time while locating the path. The selected path and chunk are printed.
If the manifest is missing, supply `--path-id ID`.

To use a shorter distance on the first held-out test path:

```bash
python run_turpy_inversion.py --step 10 --max-its 500 \
  --regularization TV --alpha 1e-3
```

For 21 z points over 4 km, `--step 10` observes the intensity at 2 km;
the default final step 20 observes it at 4 km. Use `--initial-guess rhoZ`
(default), `uniform`, or `rho0`. The last option starts from the **true**
initial intensity and is only for debugging an inverse run on a saved sample.
To select a particular path, pass `--path-id ID` using an ID from
`test_path_ids` in the split manifest. An explicit path ID overrides `--split`.

The output directory defaults to
`checkpoints/turpy_fno_4km_ic_split/inversion_path_ID_step_N/` and contains:

- `inversion.png`: true/recovered initial intensity and observed/FNO-predicted
  intensity, with absolute errors.
- `loss_history.png` and `history.json`: optimizer progress.
- `summary.json`: path, distance, and relative L2 errors.
- `inversion_results.pt`: physical-unit tensors for later analysis.

Prediction panels show SSIM and PSNR against the corresponding true image;
absolute-error panels show relative L2. These metrics also appear in
`summary.json` under `image_quality`.

The objective uses mean squared error in the checkpoint's normalized intensity
units. `--regularization` supports `None`, `L1`, `L2`, and `TV`; `--alpha`
controls its weight. `--switch-lbfgs` enables a switch from AdamW after
`--grad-patience` small-gradient steps. `--obj-tol` stops once the total loss
reaches the requested tolerance.

`--match-observed-power` enables the old script's constraint that the initial
and observed image sums match. It is off by default: zero-padded propagation
can move light outside the saved image, so the observed sum can be smaller than
the true initial sum.

This inversion measures consistency with the **FNO surrogate**. A low final
error does not by itself establish that the recovered initial intensity is
unique or matches the TurPy solution. Compare `relative_initial_l2` and
`relative_true_input_forward_l2` in `summary.json` on held-out test paths.
The latter is the FNO's prediction error when given the *true* initial
intensity, so it separates surrogate error from inversion error.

## Direct versus recursive two-step prediction

On a GPU compute node, run:

```bash
python compare_turpy_rollout.py
```

This uses the first test path from the checkpoint's `split_manifest.json` and
the first two intervals of its saved trajectory (200 m and 400 m for 21 points
over 4 km). An explicit `--path-id ID` overrides the split choice. The test
computes:

1. `rho1_hat = FNO(rho0, delta_n0, zeros..., 1/20)`.
2. `rho2_direct = FNO(rho0, delta_n0, delta_n1, zeros..., 2/20)`.
3. `rho2_recursive = FNO(rho1_hat, delta_n1, zeros..., 1/20)`.
4. A diagnostic restart with the **true** saved `rho1` in place of `rho1_hat`.

It compares every prediction to saved `rho1`/`rho2`, writes `metrics.json`,
`fields.pt`, and `comparison.png` under
`checkpoints/turpy_fno_4km_ic_split/rollout_comparison_path_ID/`, and reports
the fraction of negative pixels in `rho1_hat`. The recursive test uses the raw
FNO output as written, without clipping it to nonnegative intensity.
Prediction panels show SSIM and PSNR against the matching true image;
pointwise-error panels show relative L2 against their named reference. PSNR
uses the full intensity range of the true image, not the clipped plot colors.

The FNO was trained with the original `rho0` and accumulated screen history;
the recursive call restarts it from a propagated intensity. It is therefore a
diagnostic of reuse outside its training setup. Physical propagation also
depends on optical phase, which an intensity-only `rho1` does not contain.

## Full recursive rollout to 4 km

To repeat the restart at every propagation interval on the first held-out
test path, run:

```bash
python compare_turpy_rollout.py --full-rollout
```

At step `i`, the recursive input is the previous predicted intensity,
`delta_n[i-1]` in the first screen slot, zeros in all future slots, and a
history fraction of `1/20`. The direct input retains the true original
`rho0`, all screens through step `i`, and a fraction of `i/20`.

The script saves `full_rollout_comparison.png` with the true, direct, and
recursive final intensities, their pointwise errors, and error/SSIM curves
over distance. `full_rollout_metrics.json` contains scores at every step;
`full_rollout_fields.pt` stores the three intensity trajectories. These files
go in the same `rollout_comparison_path_ID/` directory without replacing the
two-step outputs. Use `--path-id ID` to choose another test path.

This measures how a model trained from the original `rho0` behaves when its
own predictions are fed back as new initial conditions. Recursive error can
grow because prediction errors accumulate and because intensity alone omits
the propagated optical phase.
