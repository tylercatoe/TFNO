# Autoregressive window TurPy FNO

`train_window_fno.py` trains one FNO to repeatedly predict the next intensity
image from recent intensity and refractive-index histories. It reconstructs
complete paths from the existing one-step TurPy chunks, so it works with both
turbulent and `--no-turbulence` free-space data.

## Model

Let $A(x,y,z)$ be the slowly varying envelope, $\rho=|A|^2$ the intensity,
$k_0=2 \pi/\lambda_0$, $n_0$ the background refractive index, and $\delta_n$ the
refractive-index perturbation. The generator, TurPy propagation, and inversion
use the free-space transfer function

$$
  H(f_\perp) = \text{exp}\{-i \pi \lambda_0 dz |f_\perp|^2/n_0\}.
$$

This corresponds to the envelope equation and thin-screen update

$$
  \partial _z A = \frac{i}{2 n_0 k_0} \nabla ^ 2_\perp A + i k_0 \delta_n A,\\
  A_{j+1} = \text{exp}\{i k_0 dz \delta_{n_j}\} P_{dz}(A_j).
$$

In free space, this agrees with $\nabla^2_\perp A + 2 i k \partial _z A = 0$
for $k=n_0 k_0$. If $U=A \text{exp}\{i k z\}$ is the carrier-including field,
substitution gives $\partial _z U - \frac{i}{2k} \nabla^2_\perp U - i k U = 0$.
The carrier term is a spatially uniform phase and does not change intensity.

Since intensity does not determine the field's phase, the next intensity is not
generally determined by the current intensity and screen alone.

## Autoregressive rollout

For the default 10-frame history, the model makes one prediction at each step,
as in the [original FNO-2D Navier–Stokes setup](https://arxiv.org/pdf/2010.08895):

$$
\hat\rho_{j+1}=F_\theta(\tilde\rho_{j-9},\ldots,\tilde\rho_j,
                         \delta n_{j-9},\ldots,\delta n_j),
\qquad j=9,\ldots,18,
$$

where $\tilde\rho_j=\rho_j$ for the first 10 observed frames and
$\tilde\rho_j=\hat\rho_j$ thereafter. Thus the initial $\rho_0,\ldots,\rho_9$
produce $\hat\rho_{10},\ldots,\hat\rho_{19}$ by sliding the window forward
with each prediction. The screen $\delta n_j$ is known for the interval from
$z_j$ to $z_{j+1}$. The model has 20 input channels and one output channel;
future screens are supplied as the rollout advances. In free space, all screen
channels are zero.

Training averages the objective over the 10 predicted frames and backpropagates
through the entire rollout:

$$
\mathcal L=\frac{1}{10}\sum_{j=9}^{18}
\ell(\hat\rho_{j+1},\rho_{j+1}).
$$

Validation and testing also feed predictions back, so their metrics measure
the full rollout. `relative_l2` averages framewise relative errors;
`final_step_relative_l2` reports the last frame's error.

The trainer reconstructs the sequence from the saved chunks: `Y` supplies
`rho(1)...rho(N)`, the first input's intensity channel supplies `rho(0)`, and
the active screen slots in `X` supply the corresponding `delta_n` screens.
Each path is one training example. The current 20-interval data have 21 intensity
frames and 500 paths per dataset. The default 10-step rollout uses frames
$\rho_0$ through $\rho_{19}$; set `--rollout-steps 11` to include $\rho_{20}$.
Complete paths are split by mode combination by default.

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

Set `--window` and `--rollout-steps` to change the observed history and forecast
lengths. Other options include the usual
FNO settings (`--width`, `--layers`, `--modes-y`, `--modes-x`), optimization
settings, and `--split-unit {mode-combination,path}`. The Slurm job uses a batch
size of 2 to limit memory use during backpropagation through 10 steps. Submit
either dataset from the repository directory:

```bash
sbatch slurm/train_window_fno.sbatch data/turpy_chunks_4km checkpoints/window_fno_4km_turbulent
sbatch slurm/train_window_fno.sbatch data/turpy_chunks_4km_free_space checkpoints/window_fno_4km_free_space
```

The output directory contains `best.pt`, split/config manifests, `history.json`, a
`loss_history.png` plot of training and validation loss by epoch, and test
metrics.
