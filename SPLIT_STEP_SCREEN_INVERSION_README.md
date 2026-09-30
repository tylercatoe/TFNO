# TurPy split-step screen inversion

`invert_screens_split_step.py` estimates refractive-index screens by optimizing
TurPy's differentiable coherent propagator. It uses the saved initial intensity
`rho(0)` and final intensity `rho(Z)` as observations. Saved screens are used
only after fitting, to score the estimate and check the forward model.

## Continuous model

Let `A(x,y,z)` be the slowly varying complex envelope represented by the
split-step code, `rho = |A|^2` its intensity, `lambda_0` the vacuum wavelength,
`n_0` the background refractive index, `k_0 = 2 pi / lambda_0`, and
`delta_n(x,y,z)` the atmospheric index perturbation. The current code uses the
continuous model

```text
partial_z A = +i/(2 n_0 k_0) * Laplacian_perp A
              + i k_0 delta_n A,
rho(x,y,z) = |A(x,y,z)|^2.
```

With the standard Fourier convention, this gives the Fresnel kernel
`exp(-i pi lambda_0 dz |f_perp|^2/n_0)`. This is the sign now used by the
generator, TurPy propagation, and inversion, and matches
`Laplacian_perp A + 2 i k partial_z A = 0` for `k=n_0 k_0` in free space.

Take `k = n_0 k_0`. If `A` is the envelope and the carrier-including field is
`U=A exp(i k z)`, substitution gives:

```text
partial_z U - i/(2k) Laplacian_perp U - i k U = 0
```

The `i k U` term is a spatially uniform carrier phase and does not change
intensity. In the inverse problem, screens are chosen to make the modeled
endpoint intensity match the observation:

```text
minimize_delta_n  || |A_delta_n(Z)|^2 - rho_obs(Z) ||^2 + regularization.
```

Only intensity is observed, so the initial field is taken to have zero phase.
The script requires a flat initial phase in the dataset; intensity alone cannot
specify a vortex or other unknown initial phase.

## Discrete inversion

For `N` equal intervals of length `dz = Z/N`, the script alternates a
free-space Fresnel propagation and a thin phase screen:

```text
A_{j+1} = exp(i phi_j) P_dz(A_j),
phi_j = k_0 dz delta_n_j,
rho_pred(Z) = |A_N|^2.
```

`P_dz` is the Fresnel propagator used by this TurPy code. The optimized variables are the phase
screens `phi_j` in radians; each screen has its spatial mean removed. The data
loss is relative mean-squared error in final intensity. The default smoothness
penalty discourages neighboring pixels from changing sharply; `l2` penalizes
screen magnitude, and `none` disables regularization.

The last screen is applied immediately before measuring intensity. A
phase-only screen cannot change intensity at that same plane, so the last screen
is unidentifiable from `rho(Z)`. More generally, a single endpoint intensity
does not guarantee a unique screen sequence.

## Run

On a GPU compute node, run against a generated free-space dataset like this:

```bash
python invert_screens_split_step.py \
  --data-dir data/turpy_chunks_4km_free_space \
  --path-id 0 --device cuda --max-its 1000 \
  --output-dir free_space_screen_inversion_path_0
```

For ordinary turbulent chunks, omit `--data-dir` and select a path using
`--path-id ID` (or the default split manifest). Key options include
`--learning-rate`, `--init-std`, `--regularization {none,l2,smooth}`, and
`--alpha`. The propagator defaults to zero padding with factor 2; these settings
must match the dataset generator (`--no-zero-padding` disables padding).

The output directory contains final-intensity and phase-screen plots,
`summary.json`, `history.json`, and `fields.pt`. For a free-space dataset, the
true `delta_n` screens are zero; the optimizer is still free to estimate
nonzero screens unless constrained, so compare the recovered screens with zero
as well as checking the final-intensity fit.
