# Rolling-window TurPy FNO

`train_window_fno.py` trains an FNO to predict the next intensity image from
recent intensity and refractive-index histories. It reads the existing
one-step TurPy chunks, so the same trainer works with turbulent data and with
free-space chunks generated using `--no-turbulence`.

## Model

Let `A(x,y,z)` be the slowly varying envelope, `rho=|A|^2` the intensity,
`k_0=2 pi/lambda_0`, `n_0` the background refractive index, and `delta_n` the
refractive-index perturbation. The generator, TurPy propagation, and inversion
use the free-space transfer function

```text
H(f_perp) = exp(-i pi lambda_0 dz |f_perp|^2/n_0).
```

This corresponds to the envelope equation and thin-screen update

```text
partial_z A = +i/(2 n_0 k_0) Laplacian_perp A + i k_0 delta_n A,
A_{j+1} = exp(i k_0 dz delta_n_j) P_dz(A_j).
```

In free space, this agrees with `Laplacian_perp A + 2 i k partial_z A = 0`
for `k=n_0 k_0`. If `U=A exp(i k z)` is the carrier-including field,
substitution gives `partial_z U - i/(2k) Laplacian_perp U - i k U = 0`.
The carrier term is a spatially uniform phase and does not change intensity.

Since intensity does not determine the field's phase, the next intensity is not
generally determined by the current intensity and screen alone.

This model gives the FNO a short history as context:

```text
input  = (rho[i-L+1], ..., rho[i], delta_n[i-L+1], ..., delta_n[i])
target = rho[i+1]
```

For the default `L=10`, the FNO has 20 input channels and one output channel.
For the current 20-interval trajectories, each path yields 11 windows. The
free-space version uses the same input layout with every `delta_n` channel zero.

The trainer reconstructs the sequence from the saved chunks: `Y` supplies
`rho(1)...rho(N)`, the first input's intensity channel supplies `rho(0)`, and
the active screen slots in `X` supply the corresponding `delta_n` screens.
Windows are created only after splitting by complete paths (by mode combination
by default), keeping overlapping windows from a path together.

## Train

Run from the repository directory on a GPU compute node. Choose either dataset
directory; the defaults are 4 km turbulent chunks:

```bash
python train_window_fno.py \
  --data-dir data/turpy_chunks_4km \
  --output-dir checkpoints/window_fno_4km_turbulent
```

For free-space data:

```bash
python train_window_fno.py \
  --data-dir data/turpy_chunks_4km_free_space \
  --output-dir checkpoints/window_fno_4km_free_space
```

Set `--window` to change the history length. Other options include the usual
FNO settings (`--width`, `--layers`, `--modes-y`, `--modes-x`), optimization
settings, and `--split-unit {mode-combination,path}`. The output directory
contains `best.pt`, split/config manifests, training history, and test metrics.
