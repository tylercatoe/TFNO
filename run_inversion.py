import argparse
import json
import random
from pathlib import Path
# import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
# from torch.utils.data import DataLoader

from utilities import FNO2dPropagation, FNO3dPropagation, PropagationDataset, Normalization



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/realization_ladder_ic0300_r050/best.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/fno_inversion"))
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='Learning rate in optimizer')
    parser.add_argument("--obj_tol", type=float, default=1e-4, help='Optimization loss tolerance')
    parser.add_argument("--max_its", type=int, default=1000, help="Maximum iterations in optimzation loop")
    parser.add_argument("--regularization", type=str, default="None", choices=('None', 'L1', 'L2', 'TV'), help='Objective function regularization options')
    parser.add_argument("--alpha", type=float, default=1e-3, help="Regularizatin parameter")
    parser.add_argument("--sample_folder", type=Path, default=Path('testing_inversion_code_samples'), help='Path to sample to use')
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--grad_tol", type=float, default=1e-5)
    parser.add_argument("--grad_patience", type=int, default=150)
    parser.add_argument("--switch_lbfgs", action='store_true', default=False, help='Switch from AdamW to LBFGS after small relative loss improvement and small loss gradients')
    parser.add_argument("--initial_guess", choices=("uniform", "rhoZ", "rho0"), default="rhoZ")
    return parser.parse_args()



def optimize_initial_state(
    model,
    delta_n,
    rhoZ_obs,
    rho0_initial,
    max_iterations,
    objective_tolerance,
    switch_to_lbfgs,
    grad_tol,
    grad_patience,
    learning_rate,
    normalization,
    regularization,
    alpha,
):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rho0_initial = rho0_initial.clone().detach()
    rho0_initial_phyiscal = physical(rho0_initial, normalization).clamp_min(1e-8)

    latent_rho0 = torch.log(torch.expm1(rho0_initial_phyiscal))
    latent_rho0.requires_grad_(True)

    optimizer = AdamW(
        [latent_rho0],
        lr=learning_rate,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience = 150,
        min_lr=1e-4,
        threshold=1e-3
    )

    prev_objective_val = float('inf')

    history = []
    small_gradient_steps = 0

    rhoZ_obs_physical = physical(rhoZ_obs, normalization)
    observed_power = rhoZ_obs_physical.sum(dim=(-2,-1), keepdim=True).clamp_min(1e-12)

    for iter_count in range(max_iterations):


        if prev_objective_val < objective_tolerance: 
            break

        optimizer.zero_grad()

        loss, objective_vals, _ = evaluate_loss(
            latent_rho0, 
            observed_power, 
            normalization, 
            delta_n, 
            model, 
            rhoZ_obs_physical, 
            regularization, 
            alpha
            )

        loss.backward()
        
        optimizer.step()

        # Re-evaluate after the Adam update so the logged loss and gradient
        # are measured at the same point as the LBFGS diagnostics below.
        optimizer.zero_grad()
        loss, objective_vals, _ = evaluate_loss(
            latent_rho0,
            observed_power,
            normalization,
            delta_n,
            model,
            rhoZ_obs_physical,
            regularization,
            alpha,
        )
        loss.backward()
        scheduler.step(loss.item())
        current_lr = optimizer.param_groups[0]["lr"]

        prev_objective_val = loss.item()

        grad_norm = latent_rho0.grad.detach().norm().item()
        latent_norm = latent_rho0.detach().norm().item()

        relative_grad_norm = grad_norm / max(latent_norm, 1e-12)

        if relative_grad_norm < grad_tol:
            small_gradient_steps += 1
        else:
            small_gradient_steps = 0

        if iter_count % 100 == 0:
            print(f'iteration: {iter_count}  |   data loss: {objective_vals["data_loss_l2"].item():.3e}  |   LR: {current_lr:.3e}')

        if regularization != "None":
            history.append(
                {
                    "iteration": iter_count,
                    "optimizer": "AdamW",
                    "total_loss": loss.item(),
                    "data_loss": objective_vals["data_loss_l2"].item(),
                    "regularization_loss": objective_vals["reg_loss"].item(),
                    "learning_rate": current_lr,
                    "gradient_norm": grad_norm,
                    "relative_gradient_norm": relative_grad_norm,
                }
            )
        else:
            history.append(
                {
                    "iteration": iter_count,
                    "optimizer": "AdamW",
                    "total_loss": loss.item(),
                    "data_loss": objective_vals["data_loss_l2"].item(),
                    "learning_rate": current_lr,
                    "gradient_norm": grad_norm,
                    "relative_gradient_norm": relative_grad_norm,
                }
            )

        if small_gradient_steps >= grad_patience and iter_count + 1 >= 500:
            if switch_to_lbfgs:
                print(f"Switching to LBFGS after iteration {iter_count + 1}...")

                lbfgs = torch.optim.LBFGS(
                    [latent_rho0],
                    lr=1.0,
                    max_iter=1,
                    max_eval=20,
                    history_size=20,
                    line_search_fn="strong_wolfe",
                )
                lbfgs_small_gradient_steps = 0

                remaining_lbfgs_steps = max_iterations - (iter_count + 1)
                for lbfgs_iter in range(remaining_lbfgs_steps):
                    def closure():
                        lbfgs.zero_grad()
                        lbfgs_loss, _, _ = evaluate_loss(
                            latent_rho0,
                            observed_power,
                            normalization,
                            delta_n,
                            model,
                            rhoZ_obs_physical,
                            regularization,
                            alpha,
                        )
                        lbfgs_loss.backward()
                        return lbfgs_loss

                    lbfgs.step(closure)

                    lbfgs.zero_grad()
                    loss, objective_vals, _ = evaluate_loss(
                        latent_rho0,
                        observed_power,
                        normalization,
                        delta_n,
                        model,
                        rhoZ_obs_physical,
                        regularization,
                        alpha,
                    )
                    loss.backward()

                    grad_norm = latent_rho0.grad.detach().norm().item()
                    latent_norm = latent_rho0.detach().norm().item()
                    relative_grad_norm = grad_norm / max(latent_norm, 1e-12)
                    current_lr = lbfgs.state[latent_rho0].get("t", float("nan"))
                    if torch.is_tensor(current_lr):
                        current_lr = current_lr.detach().cpu().item()
                    else:
                        current_lr = float(current_lr)

                    if relative_grad_norm < grad_tol:
                        lbfgs_small_gradient_steps += 1
                    else:
                        lbfgs_small_gradient_steps = 0

                    lbfgs_history = {
                        "iteration": iter_count + lbfgs_iter + 1,
                        "optimizer": "LBFGS",
                        "total_loss": loss.item(),
                        "data_loss": objective_vals["data_loss_l2"].item(),
                        "learning_rate": current_lr,
                        "gradient_norm": grad_norm,
                        "relative_gradient_norm": relative_grad_norm,
                    }
                    if regularization != "None":
                        lbfgs_history["regularization_loss"] = objective_vals["reg_loss"].item()
                    history.append(lbfgs_history)

                    if lbfgs_iter % 25 == 0:
                        print(
                            f"LBFGS iteration: {iter_count + lbfgs_iter + 1}  |  "
                            f"data loss: {objective_vals['data_loss_l2'].item():.3e}"
                        )

                    if lbfgs_small_gradient_steps >= grad_patience:
                        break

            break


    with torch.no_grad():
        positive_rho0 = torch.nn.functional.softplus(latent_rho0) + 1e-8
        rho0_physical = positive_rho0 * observed_power / positive_rho0.sum(dim=(-2, -1), keepdim=True)

    return rho0_physical.detach(), history


def evaluate_loss(latent_rho0, observed_power, normalization, delta_n, model, rhoZ_obs_physical, regularization, alpha):
    rho0_physical = torch.nn.functional.softplus(latent_rho0) + 1e-8
    rho0_physical = rho0_physical * observed_power / rho0_physical.sum(dim=(-2,-1), keepdim=True)

    rho0_normalized = (rho0_physical - normalization.rho_mean) / normalization.rho_std
    
    rhoZ_iter = model(rho0_normalized, delta_n)
    rhoZ_iter_physical = physical(rhoZ_iter, normalization)
    objective_vals = objective_func(rhoZ_iter_physical, rhoZ_obs_physical, regularization, rho0_physical)

    loss = 0.5 * objective_vals['data_loss_l2']
    if regularization != "None":
        loss += torch.tensor(alpha) / 2 * objective_vals['reg_loss']

    return loss, objective_vals, rho0_physical

def physical(field: torch.Tensor, normalization: Normalization) -> torch.Tensor:
    return field * normalization.rho_std + normalization.rho_mean

def objective_func(rhoZ_iter, rhoZ_obs, regularization, prev_rho0) -> dict:
    data_loss_l2 = torch.sum((rhoZ_iter - rhoZ_obs).square()) / (rhoZ_iter.shape[0] **2)
    if regularization == 'None':
        losses = {'data_loss_l2': data_loss_l2}
    elif regularization == 'L1':
        reg_loss = torch.sum(torch.abs(prev_rho0)) / (prev_rho0.shape[0] **2)
        losses = {
            'data_loss_l2': data_loss_l2,
            'reg_loss': reg_loss
        }
    elif regularization == 'L2':
        reg_loss = torch.sum((prev_rho0).square()) / (prev_rho0.shape[0] **2)
        losses = {
            'data_loss_l2': data_loss_l2,
            'reg_loss': reg_loss
        }
    elif regularization == 'TV':
        dx = prev_rho0[..., :, 1:] - prev_rho0[..., :, :-1]
        dy = prev_rho0[..., 1:, :] - prev_rho0[..., :-1, :]

        # Anisotropic TV
        reg_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        losses = {
            'data_loss_l2': data_loss_l2,
            'reg_loss': reg_loss
        }
    else:
        raise ValueError(f'Invaid regularization passed: {regularization}')

    return losses

def load_model(checkpoint) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_class = FNO2dPropagation if checkpoint.get("model_type", "3d") == "2d" else FNO3dPropagation
    model = model_class(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

def save_plot(path, rhoZ, rho0_true, rho0_pred, rhoZ_pred, data_loss=None):
    rhoZ = rhoZ.detach().cpu().numpy()
    rho0_true = rho0_true.detach().cpu().numpy()
    rho0_pred = rho0_pred.detach().cpu().numpy()
    rhoZ_pred = rhoZ_pred.detach().cpu().numpy()
    

    rho0_error = abs(rho0_pred - rho0_true)
    rhoZ_error = abs(rhoZ_pred - rhoZ)

    rho0_values = np.concatenate([
        rho0_true.ravel(),
        rho0_pred.ravel(),
    ])

    rhoZ_values = np.concatenate([
        rhoZ.ravel(),
        rhoZ_pred.ravel(),
    ])

    rho0_vmax = np.percentile(rho0_values, 99)
    rhoZ_vmax = np.percentile(rhoZ_values, 99)

    rho0_error_vmax = np.percentile(rho0_error.ravel(), 99)
    rhoZ_error_vmax = np.percentile(rhoZ_error.ravel(), 99)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    images = (
        axes[0, 0].imshow(rho0_true, vmin=0, vmax=rho0_vmax, cmap="magma"),
        axes[0, 1].imshow(rho0_pred, vmin=0, vmax=rho0_vmax, cmap="magma"),
        axes[0, 2].imshow(rho0_error, vmin=0, vmax=rho0_error_vmax, cmap="magma"),

        axes[1, 0].imshow(rhoZ, vmin=0, vmax=rhoZ_vmax, cmap="magma"),
        axes[1, 1].imshow(rhoZ_pred, vmin=0, vmax=rhoZ_vmax, cmap="magma"),
        axes[1, 2].imshow(rhoZ_error, vmin=0, vmax=rhoZ_error_vmax, cmap="magma"),
    )
    titles = (
        r"$\rho(x,0)$ True",
        r"$\widehat{\rho}(x,0)$ Prediction",
        "Absolute initial error",
        r"$\rho(x,Z)$ True",
        r"$\widehat{\rho}(x,Z)$ Prediction",
        "Absolute final error",
    )
    if data_loss is not None:
        titles = (*titles[:-1], f"Absolute final error\nL2 data loss: {data_loss:.3e}")

    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=20)
        axis.set_axis_off()
    for axis, image in zip(axes.flat, images):
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.savefig(path, dpi=160)
    plt.close(figure)

def save_loss_plot(path, history):
    """Save the inversion loss history with a logarithmic y-axis."""
    if not history:
        return

    iterations = [entry["iteration"] for entry in history]
    data_loss = [max(entry["data_loss"], 1.0e-30) for entry in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    loss_axis, gradient_axis = axes
    loss_axis.plot(iterations, data_loss, label="Data loss")

    if "regularization_loss" in history[0]:
        regularization_loss = [
            max(entry["regularization_loss"], 1.0e-30)
            for entry in history
        ]
        total_loss = [max(entry["total_loss"], 1.0e-30) for entry in history]
        loss_axis.plot(iterations, regularization_loss, label="Regularization loss")
        loss_axis.plot(iterations, total_loss, label="Total loss")

    loss_axis.set_yscale("log")
    loss_axis.set_xlabel("Iteration")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Inversion losses")
    loss_axis.grid(True, which="both", alpha=0.3)
    loss_axis.legend()

    gradient_norms = [entry.get("gradient_norm") for entry in history]
    relative_gradient_norms = [
        entry.get("relative_gradient_norm") for entry in history
    ]
    if all(value is not None and value > 0 for value in gradient_norms):
        gradient_axis.plot(
            iterations,
            [max(value, 1.0e-30) for value in gradient_norms],
            label="Gradient norm",
        )
        if all(value is not None and value > 0 for value in relative_gradient_norms):
            gradient_axis.plot(
                iterations,
                [max(value, 1.0e-30) for value in relative_gradient_norms],
                label="Relative gradient norm",
            )
        gradient_axis.set_yscale("log")
        gradient_axis.set_ylabel("Gradient norm")
        gradient_axis.legend()
    else:
        gradient_axis.text(
            0.5,
            0.5,
            "Gradient history unavailable",
            ha="center",
            va="center",
            transform=gradient_axis.transAxes,
        )
    gradient_axis.set_xlabel("Iteration")
    gradient_axis.set_title("Gradient norms")
    gradient_axis.grid(True, which="both", alpha=0.3)

    figure.savefig(path, dpi=160)
    plt.close(figure)
    

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    normalization = Normalization(**checkpoint["normalization"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or args.checkpoint.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint=checkpoint).to(device)
    sample_files = sorted(Path(args.sample_folder).glob("*.pt"))
    dataset = PropagationDataset(sample_files, normalization, checkpoint['resolution'])

    rho_0s = []
    histories = []
    for i in range(len(dataset)):
        sample = dataset[i]

        rhoZ_obs_physical = physical(sample["target"], normalization)

        if args.initial_guess == 'uniform':

            observed_power = rhoZ_obs_physical.sum().clamp_min(1e-12)

            rho0_guess_physical = torch.full_like(
                rhoZ_obs_physical,
                observed_power / rhoZ_obs_physical.numel(),
            )

            rho0_guess = (
                rho0_guess_physical - normalization.rho_mean
            ) / normalization.rho_std
            rho0_guess = rho0_guess.unsqueeze(0).to(device)

        elif args.initial_guess == 'rhoZ':
            rho0_guess = sample['target'].unsqueeze(0).to(device)

        elif args.initial_guess == 'rho0':
            rho0_guess = sample['rho0'].unsqueeze(0).to(device)


        rho0, hist = optimize_initial_state(
            model=model,
            delta_n=sample["delta_n"].unsqueeze(0).to(device),
            rhoZ_obs=sample["target"].unsqueeze(0).to(device),
            rho0_initial=rho0_guess,
            max_iterations=args.max_its,
            objective_tolerance=args.obj_tol,
            switch_to_lbfgs=args.switch_lbfgs,
            grad_tol=args.grad_tol,
            grad_patience=args.grad_patience,
            learning_rate=args.learning_rate,
            normalization=normalization,
            regularization=args.regularization,
            alpha=args.alpha
        )
        final_rho_Z_predicted = model(((rho0 - normalization.rho_mean) / normalization.rho_std), sample["delta_n"].unsqueeze(0).to(device))
        save_plot(
            output_dir / f"sample{i}.png",
            physical(sample["target"], normalization),
            physical(sample["rho0"], normalization),
            rho0.squeeze(0),
            physical(final_rho_Z_predicted, normalization).squeeze(0),
            data_loss=hist[-1]["data_loss"],
        )
        save_loss_plot(output_dir / f"loss_history_sample_{i}.png", hist)
        rho_0s.append(rho0.cpu())
        histories.append(hist)
        print(
            f"Finished sample {i}: "
            f"it_count: {hist[-1]['iteration']}, "
            f"data loss: {hist[-1]['data_loss']}"
        )
        with (output_dir / f"history_sample_{i}.json").open("w") as file:
            json.dump(hist, file, indent=2)

    torch.save(
        {
            "rho0_physical": rho_0s,
            "histories": histories,
            "checkpoint": str(args.checkpoint),
            "sample_folder": args.sample_folder,
        },
        output_dir / "inversion_results.pt",
    )



if __name__ == '__main__':
    main()
