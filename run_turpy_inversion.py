"""Recover an initial intensity from a TurPy FNO prediction at one z step."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from inversion_evaluation import (
    append_path_metrics, begin_test_set_run, iter_final_test_examples,
    save_rho0_test_set_plot, summarize_rho0_records, test_path_ids,
)
from utilities import FNO2d, Normalization, image_comparison_metrics, load_turpy_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/turpy_fno_4km_ic_split/best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/turpy_chunks_4km"))
    parser.add_argument("--chunk-pattern", default="chunk_*.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--path-id", type=int, default=None)
    parser.add_argument("--all-test-paths", action="store_true", help="Invert rho0 once per held-out test path at final z.")
    parser.add_argument("--max-paths", type=int, help="Limit test paths for a smoke test.")
    parser.add_argument("--example-plots", type=int, default=3, help="Number of paths with detailed plots/fields.")
    parser.add_argument("--resume", action="store_true", help="Continue a test-set run using existing metric rows.")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Target z-grid step, 1..n_z-1; defaults to the final step.",
    )
    parser.add_argument("--initial-guess", choices=("rhoZ", "rho0", "uniform"), default="rhoZ")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-its", type=int, default=1000)
    parser.add_argument("--obj-tol", type=float, default=1e-4)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--grad-patience", type=int, default=150)
    parser.add_argument("--switch-lbfgs", action="store_true")
    parser.add_argument("--regularization", choices=("None", "L1", "L2", "TV"), default="None")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument(
        "--match-observed-power",
        action="store_true",
        help="Constrain sum(rho0) to sum(observed intensity); usually avoid with cropped beams.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_path_id(args: argparse.Namespace) -> int:
    if args.path_id is not None:
        return args.path_id
    manifest_path = args.checkpoint.parent / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; provide --path-id or copy the split manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest[f"{args.split}_path_ids"]
    if not selected:
        raise ValueError(f"No paths in the {args.split} split")
    return int(selected[0])


def load_observation(
    data_dir: Path, pattern: str, path_id: int, step: int | None
) -> tuple[torch.Tensor, torch.Tensor, int, int, float, Path]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No chunks matching {pattern!r} in {data_dir}")
    for path in paths:
        chunk = load_turpy_file(path)
        path_ids = chunk["path_ids"]
        selected_path = torch.nonzero(path_ids == path_id, as_tuple=True)[0]
        if selected_path.numel() == 0:
            del chunk
            continue
        n_intervals = int(chunk["n_z"]) - 1
        selected_step = n_intervals if step is None else step
        if not 1 <= selected_step <= n_intervals:
            raise ValueError(f"--step must be between 1 and {n_intervals}")
        horizon_values = chunk["X"].index_select(0, selected_path)[:, 0, 0, -1]
        steps = torch.round(horizon_values * n_intervals).long()
        matches = selected_path[steps == selected_step]
        if matches.numel() != 1:
            raise ValueError(
                f"Expected one example for path {path_id}, step {selected_step} in {path}"
            )
        index = int(matches[0])
        x = chunk["X"][index].float().clone()
        y = chunk["Y"][index].float().clone()
        total_distance = float(chunk["total_distance"])
        del chunk
        return x, y, selected_step, n_intervals, total_distance, path
    raise ValueError(f"Path ID {path_id} was not found in {data_dir}")


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for a strictly positive starting image."""
    return value + torch.log(-torch.expm1(-value))


def initial_intensity(
    latent: torch.Tensor, observed: torch.Tensor, match_observed_power: bool
) -> torch.Tensor:
    rho0 = F.softplus(latent) + 1e-8
    if match_observed_power:
        rho0 = rho0 * observed.sum() / rho0.sum().clamp_min(1e-12)
    return rho0


def objective(
    latent: torch.Tensor,
    fixed_input: torch.Tensor,
    observed: torch.Tensor,
    model: FNO2d,
    normalization: Normalization,
    regularization: str,
    alpha: float,
    match_observed_power: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    rho0 = initial_intensity(latent, observed, match_observed_power)
    normalized_rho0 = normalization.normalize_target(rho0)
    model_input = torch.cat((normalized_rho0, fixed_input[..., 1:]), dim=-1)
    prediction_normalized = model(model_input)
    target_normalized = normalization.normalize_target(observed)
    data_loss = (prediction_normalized - target_normalized).square().mean()

    # Express penalties in intensity-std units, consistent with the data loss.
    scaled_rho0 = rho0 / normalization.intensity_std
    if regularization == "L1":
        reg_loss = scaled_rho0.abs().mean()
    elif regularization == "L2":
        reg_loss = scaled_rho0.square().mean()
    elif regularization == "TV":
        dx = scaled_rho0[:, :, 1:, :] - scaled_rho0[:, :, :-1, :]
        dy = scaled_rho0[:, 1:, :, :] - scaled_rho0[:, :-1, :, :]
        reg_loss = dx.abs().mean() + dy.abs().mean()
    else:
        reg_loss = data_loss.new_zeros(())

    total_loss = 0.5 * (data_loss + alpha * reg_loss)
    metrics = {"data_loss": data_loss, "regularization_loss": reg_loss}
    return total_loss, metrics, rho0, normalization.denormalize_target(prediction_normalized)


def optimize_initial_state(
    model: FNO2d,
    fixed_input: torch.Tensor,
    observed: torch.Tensor,
    guess: torch.Tensor,
    normalization: Normalization,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    # Avoid nearly zero softplus derivatives where the observed image is dark.
    floor = max(1e-8, 1e-3 * normalization.intensity_std)
    latent = inverse_softplus(guess.clamp_min(floor)).detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([latent], lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=150, min_lr=1e-5
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_latent = latent.detach().clone()
    small_gradient_steps = 0
    lbfgs = None

    for iteration in range(1, args.max_its + 1):
        active_optimizer = optimizer if lbfgs is None else lbfgs

        def closure() -> torch.Tensor:
            active_optimizer.zero_grad()
            loss, _, _, _ = objective(
                latent, fixed_input, observed, model, normalization,
                args.regularization, args.alpha, args.match_observed_power,
            )
            loss.backward()
            return loss

        if lbfgs is None:
            closure()
            optimizer.step()
        else:
            lbfgs.step(closure)

        active_optimizer.zero_grad()
        loss, components, _, _ = objective(
            latent, fixed_input, observed, model, normalization,
            args.regularization, args.alpha, args.match_observed_power,
        )
        loss.backward()
        current_loss = float(loss.detach())
        grad_norm = float(latent.grad.detach().norm())
        relative_grad_norm = grad_norm / max(float(latent.detach().norm()), 1e-12)
        if lbfgs is None:
            scheduler.step(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_latent = latent.detach().clone()

        history.append({
            "iteration": iteration,
            "optimizer": "AdamW" if lbfgs is None else "LBFGS",
            "total_loss": current_loss,
            "data_loss": float(components["data_loss"].detach()),
            "regularization_loss": float(components["regularization_loss"].detach()),
            "learning_rate": active_optimizer.param_groups[0]["lr"],
            "gradient_norm": grad_norm,
            "relative_gradient_norm": relative_grad_norm,
        })
        if iteration == 1 or iteration % 100 == 0:
            print(
                f"iteration {iteration:04d} | data={history[-1]['data_loss']:.4e} "
                f"| total={current_loss:.4e} | optimizer={history[-1]['optimizer']}",
                flush=True,
            )
        if current_loss <= args.obj_tol:
            break

        small_gradient_steps = (
            small_gradient_steps + 1 if relative_grad_norm < args.grad_tol else 0
        )
        if small_gradient_steps >= args.grad_patience:
            if args.switch_lbfgs and lbfgs is None:
                lbfgs = torch.optim.LBFGS(
                    [latent], lr=1.0, max_iter=1, max_eval=20,
                    history_size=20, line_search_fn="strong_wolfe",
                )
                small_gradient_steps = 0
                print(f"Switching to LBFGS after iteration {iteration}", flush=True)
            else:
                break

    with torch.no_grad():
        rho0 = initial_intensity(best_latent, observed, args.match_observed_power)
        model_input = torch.cat(
            (normalization.normalize_target(rho0), fixed_input[..., 1:]), dim=-1
        )
        prediction = normalization.denormalize_target(model(model_input))
    return rho0.detach(), prediction.detach(), history


def save_image_plot(
    path: Path,
    rho0_true: torch.Tensor,
    rho0_est: torch.Tensor,
    observed: torch.Tensor,
    predicted: torch.Tensor,
    z_m: float,
    quality: dict[str, dict[str, float]],
) -> None:
    images = [rho0_true, rho0_est, (rho0_est - rho0_true).abs(),
              observed, predicted, (predicted - observed).abs()]
    images = [image.squeeze().detach().cpu() for image in images]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    titles = [
        "True rho(0)",
        f"Recovered rho(0)\nSSIM={quality['initial']['ssim']:.3f}  PSNR={quality['initial']['psnr_db']:.2f} dB",
        f"Absolute initial error\nrel L2={quality['initial']['relative_l2']:.3e}",
        f"Observed rho({z_m:g} m)",
        f"FNO prediction\nSSIM={quality['final']['ssim']:.3f}  PSNR={quality['final']['psnr_db']:.2f} dB",
        f"Absolute final error\nrel L2={quality['final']['relative_l2']:.3e}",
    ]
    for row in range(2):
        scale = max(float(torch.quantile(torch.cat((images[3 * row], images[3 * row + 1])).flatten(), 0.99)), 1e-12)
        for column in range(3):
            index = 3 * row + column
            vmax = scale if column < 2 else max(float(torch.quantile(images[index].flatten(), 0.99)), 1e-12)
            image = axes[row, column].imshow(images[index], vmin=0, vmax=vmax, cmap="magma")
            axes[row, column].set_title(titles[index], fontsize=11)
            axes[row, column].set_axis_off()
            figure.colorbar(image, ax=axes[row, column], shrink=0.75)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_loss_plot(path: Path, history: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    iterations = [item["iteration"] for item in history]
    loss_keys = ["data_loss", "total_loss"]
    if any(item.get("regularization_loss", 0.0) > 0 for item in history):
        loss_keys.insert(1, "regularization_loss")
    for key in loss_keys:
        axes[0].plot(iterations, [max(item[key], 1e-30) for item in history], label=key)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for key in ("gradient_norm", "relative_gradient_norm"):
        axes[1].plot(iterations, [max(item[key], 1e-30) for item in history], label=key)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def invert_one_path(
    model: FNO2d, normalization: Normalization, x: torch.Tensor,
    y: torch.Tensor, step: int, n_intervals: int, total_distance: float,
    chunk_path: Path, path_id: int, args: argparse.Namespace,
    output_dir: Path | None, mode_combination_index: int | None = None,
) -> dict:
    """Invert one initial intensity; write full artifacts only when requested."""
    device = next(model.parameters()).device
    path_seed = args.seed + path_id if args.all_test_paths else args.seed
    random.seed(path_seed)
    torch.manual_seed(path_seed)
    if x.shape[-1] != model.input_channels:
        raise ValueError(
            f"Checkpoint expects {model.input_channels} channels but sample has {x.shape[-1]}"
        )
    if y.ndim == 2:
        y = y.unsqueeze(-1)
    observed = y.unsqueeze(0).to(device)
    rho0_true = x[..., :1].unsqueeze(0).to(device)
    fixed_input = x.unsqueeze(0).to(device).clone()
    fixed_input[..., 1:-1] /= normalization.delta_n_rms
    if args.initial_guess == "rhoZ":
        guess = observed.clone()
    elif args.initial_guess == "rho0":
        guess = rho0_true.clone()
    else:
        guess = torch.full_like(observed, float(observed.mean()))

    z_m = total_distance * step / n_intervals
    print(f"Inverting path {path_id}, step {step}/{n_intervals}, z={z_m:g} m from {chunk_path}")
    recovered, predicted, history = optimize_initial_state(
        model, fixed_input, observed, guess, normalization, args
    )
    with torch.no_grad():
        true_input = torch.cat(
            (normalization.normalize_target(rho0_true), fixed_input[..., 1:]), dim=-1
        )
        true_input_prediction = normalization.denormalize_target(model(true_input))
    relative_true_input_forward = float(
        (true_input_prediction - observed).norm() / observed.norm().clamp_min(1e-12)
    )
    relative_final = float((predicted - observed).norm() / observed.norm().clamp_min(1e-12))
    relative_initial = float((recovered - rho0_true).norm() / rho0_true.norm().clamp_min(1e-12))
    relative_guess = float((guess - rho0_true).norm() / rho0_true.norm().clamp_min(1e-12))
    quality = {
        "initial": image_comparison_metrics(recovered, rho0_true),
        "final": image_comparison_metrics(predicted, observed),
    }
    metadata = {
        "checkpoint": str(args.checkpoint),
        "chunk": str(chunk_path),
        "path_id": path_id,
        "mode_combination_index": mode_combination_index,
        "step": step,
        "z_m": z_m,
        "initial_guess": args.initial_guess,
        "match_observed_power": args.match_observed_power,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "iterations": len(history),
        "best_total_loss": min(item["total_loss"] for item in history),
        "relative_final_l2": relative_final,
        "relative_initial_l2": relative_initial,
        "relative_guess_l2": relative_guess,
        "relative_true_input_forward_l2": relative_true_input_forward,
        "image_quality": quality,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image_plot(
            output_dir / "inversion.png", rho0_true, recovered,
            observed, predicted, z_m, quality,
        )
        save_loss_plot(output_dir / "loss_history.png", history)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save({
            "rho0_physical": recovered.cpu(),
            "rho0_true_physical": rho0_true.cpu(),
            "rho_target_physical": observed.cpu(),
            "rho_prediction_physical": predicted.cpu(),
            "rho_true_input_prediction_physical": true_input_prediction.cpu(),
            **metadata,
        }, output_dir / "inversion_results.pt")
    print(
        f"Done: final relative L2={relative_final:.4e}, "
        f"initial relative L2={relative_initial:.4e}, "
        f"FNO error with true rho0={relative_true_input_forward:.4e}",
        flush=True,
    )
    return metadata


def main() -> None:
    args = parse_args()
    if args.max_its < 1 or args.grad_patience < 1 or args.learning_rate <= 0:
        raise ValueError("--max-its, --grad-patience, and --learning-rate must be positive")
    if args.obj_tol < 0 or args.grad_tol < 0 or args.alpha < 0 or args.example_plots < 0:
        raise ValueError("Tolerances, --alpha, and --example-plots cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    checkpoint = load_checkpoint(args.checkpoint, device)
    normalization = Normalization.from_state_dict(checkpoint["normalization"])
    model = FNO2d(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not args.all_test_paths:
        if args.resume or args.max_paths is not None:
            raise ValueError("--resume and --max-paths require --all-test-paths")
        path_id = choose_path_id(args)
        x, y, step, n_intervals, distance, chunk_path = load_observation(
            args.data_dir, args.chunk_pattern, path_id, args.step
        )
        output_dir = args.output_dir or args.checkpoint.parent / f"inversion_path_{path_id}_step_{step}"
        invert_one_path(model, normalization, x, y, step, n_intervals, distance,
                        chunk_path, path_id, args, output_dir)
        return
    if args.path_id is not None or args.split != "test" or args.step is not None:
        raise ValueError("--all-test-paths uses the test split at final z; omit --path-id, --split, and --step")
    if args.initial_guess == "rho0":
        raise ValueError("--initial-guess rho0 uses the truth and is not valid for test-set evaluation")
    manifest = args.checkpoint.parent / "split_manifest.json"
    selected = test_path_ids(manifest, args.max_paths)
    output_dir = args.output_dir or args.checkpoint.parent / "rho0_inversion_test_set"
    config = {"method": "fno_rho0", "checkpoint": str(args.checkpoint.resolve()),
              "data_dir": str(args.data_dir.resolve()), "chunk_pattern": args.chunk_pattern,
              "max_its": args.max_its, "learning_rate": args.learning_rate,
              "initial_guess": args.initial_guess, "obj_tol": args.obj_tol,
              "grad_tol": args.grad_tol, "grad_patience": args.grad_patience,
              "switch_lbfgs": args.switch_lbfgs, "regularization": args.regularization,
              "alpha": args.alpha, "match_observed_power": args.match_observed_power,
              "seed": args.seed, "device": str(device)}
    records, rows_path = begin_test_set_run(output_dir, config, args.resume)
    completed = {row["path_id"] for row in records}
    if completed - set(selected):
        raise ValueError("Existing metrics include paths outside the selected test set")
    example_ids = set(selected[:args.example_plots])
    for path_id, x, y, metadata, chunk_path in iter_final_test_examples(
        args.data_dir, args.chunk_pattern, selected
    ):
        if path_id in completed:
            continue
        example_dir = output_dir / "examples" / f"path_{path_id}" if path_id in example_ids else None
        record = invert_one_path(
            model, normalization, x, y, metadata["n_intervals"],
            metadata["n_intervals"], metadata["total_distance"],
            chunk_path, path_id, args, example_dir, metadata["mode_combination_index"],
        )
        append_path_metrics(rows_path, record)
        records.append(record)
        print(f"Completed {len(records)}/{len(selected)} held-out rho0 inversions", flush=True)
    summary = summarize_rho0_records(records)
    summary["method"] = "fno_rho0"
    (output_dir / "test_set_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_rho0_test_set_plot(output_dir / "test_set_summary.png", records)
    print(f"Test set complete: {len(records)} paths; summary: {output_dir / 'test_set_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
