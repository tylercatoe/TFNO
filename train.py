"""Train a 2D FNO to predict the next TurPy intensity field."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utilities import (
    ChunkedTurpyDataset,
    ChunkShuffleSampler,
    FNO2d,
    Normalization,
    compute_chunked_normalization,
    mode_combination_ids_for_paths,
    scan_turpy_chunks,
    split_path_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D FNO on chunked TurPy one-step examples."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("turpy_chunks"))
    parser.add_argument("--chunk-pattern", default="*.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/turpy_fno"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes-y", type=int, default=16)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--no-skip", dest="use_skip", action="store_false", default=True)
    parser.add_argument("--loss", choices=("mse", "relative-l2", "blend"), default="blend")
    parser.add_argument("--blend-mse-weight", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=("cosine", "plateau"), default="cosine")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-unit",
        choices=("mode-combination", "path"),
        default="mode-combination",
        help=(
            "Group all turbulence realizations of an initial mode combination "
            "in one split, or split independent paths directly."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalization-batch-size", type=int, default=16)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("turpy_fno_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "training.log", mode="w")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def relative_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    prediction = normalization.denormalize_target(prediction)
    target = normalization.denormalize_target(target)
    error = (prediction - target).flatten(1).norm(dim=1)
    target_norm = target.flatten(1).norm(dim=1).clamp_min(1.0e-12)
    return error / target_norm


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    return relative_l2_per_sample(prediction, target, normalization).mean()


def objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    blend_mse_weight: float,
    normalization: Normalization,
) -> torch.Tensor:
    normalized_mse = torch.mean((prediction - target).square())
    if loss_type == "mse":
        return normalized_mse
    relative = relative_l2(prediction, target, normalization)
    if loss_type == "relative-l2":
        return relative
    return relative + blend_mse_weight * normalized_mse


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    blend_mse_weight: float,
    normalization: Normalization,
    total_distance: float,
) -> dict:
    model.eval()
    totals = {
        "loss": 0.0,
        "normalized_mse": 0.0,
        "physical_mse": 0.0,
        "relative_l2": 0.0,
        "rho0_baseline_physical_mse": 0.0,
        "rho0_baseline_relative_l2": 0.0,
    }
    horizon_totals: dict[int, dict[str, float]] = {}
    sample_count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        prediction = model(x)
        batch_size = x.shape[0]
        normalized_mse_samples = (prediction - target).square().flatten(1).mean(dim=1)
        prediction_physical = normalization.denormalize_target(prediction)
        target_physical = normalization.denormalize_target(target)
        physical_mse_samples = (
            (prediction_physical - target_physical).square().flatten(1).mean(dim=1)
        )
        relative_samples = relative_l2_per_sample(prediction, target, normalization)

        rho0_baseline = x[..., 0].unsqueeze(-1)
        baseline_physical = normalization.denormalize_target(rho0_baseline)
        baseline_mse_samples = (
            (baseline_physical - target_physical).square().flatten(1).mean(dim=1)
        )
        baseline_relative_samples = relative_l2_per_sample(
            rho0_baseline, target, normalization
        )

        if loss_type == "mse":
            loss_samples = normalized_mse_samples
        elif loss_type == "relative-l2":
            loss_samples = relative_samples
        else:
            loss_samples = relative_samples + blend_mse_weight * normalized_mse_samples

        totals["loss"] += loss_samples.sum().item()
        totals["normalized_mse"] += normalized_mse_samples.sum().item()
        totals["physical_mse"] += physical_mse_samples.sum().item()
        totals["relative_l2"] += relative_samples.sum().item()
        totals["rho0_baseline_physical_mse"] += baseline_mse_samples.sum().item()
        totals["rho0_baseline_relative_l2"] += baseline_relative_samples.sum().item()

        n_intervals = x.shape[-1] - 2
        horizon_steps = torch.round(x[:, 0, 0, -1] * n_intervals).long()
        for step in torch.unique(horizon_steps).tolist():
            selected = horizon_steps == step
            entry = horizon_totals.setdefault(
                int(step),
                {
                    "count": 0.0,
                    "normalized_mse": 0.0,
                    "physical_mse": 0.0,
                    "relative_l2": 0.0,
                    "rho0_baseline_physical_mse": 0.0,
                    "rho0_baseline_relative_l2": 0.0,
                },
            )
            entry["count"] += int(selected.sum())
            entry["normalized_mse"] += normalized_mse_samples[selected].sum().item()
            entry["physical_mse"] += physical_mse_samples[selected].sum().item()
            entry["relative_l2"] += relative_samples[selected].sum().item()
            entry["rho0_baseline_physical_mse"] += baseline_mse_samples[selected].sum().item()
            entry["rho0_baseline_relative_l2"] += baseline_relative_samples[selected].sum().item()
        sample_count += batch_size
    metrics = {name: value / sample_count for name, value in totals.items()}
    metrics["per_horizon"] = []
    for step in sorted(horizon_totals):
        entry = horizon_totals[step]
        count = int(entry["count"])
        metrics["per_horizon"].append(
            {
                "step": step,
                "z_m": total_distance * step / n_intervals,
                "count": count,
                **{
                    name: value / count
                    for name, value in entry.items()
                    if name != "count"
                },
            }
        )
    return metrics


def save_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def save_prediction_diagnostics(
    model: nn.Module,
    dataset: ChunkedTurpyDataset,
    path_id: int,
    normalization: Normalization,
    total_distance: float,
    device: torch.device,
    output_path: Path,
) -> None:
    """Plot targets, best-checkpoint predictions, and errors at three ranges."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_intervals = dataset.entries[0][0].n_z - 1
    requested_steps = sorted({1, max(1, n_intervals // 2), n_intervals})
    examples: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    model.eval()
    for index in range(len(dataset)):
        sample = dataset[index]
        if int(sample["path_id"]) != path_id:
            continue
        step = int(round(float(sample["x"][0, 0, -1]) * n_intervals))
        if step not in requested_steps:
            continue
        prediction = model(sample["x"].unsqueeze(0).to(device))[0].cpu()
        prediction = normalization.denormalize_target(prediction)[..., 0]
        target = normalization.denormalize_target(sample["y"])[..., 0]
        examples[step] = (target, prediction)
        if len(examples) == len(requested_steps):
            break
    dataset.clear_cache()
    missing = set(requested_steps) - examples.keys()
    if missing:
        raise RuntimeError(
            f"Could not find diagnostic steps {sorted(missing)} for path {path_id}"
        )

    figure, axes = plt.subplots(3, len(requested_steps), figsize=(12, 10), squeeze=False)
    for column, step in enumerate(requested_steps):
        target, prediction = examples[step]
        error = (prediction - target).abs()
        intensity_values = torch.cat((target.flatten(), prediction.flatten()))
        intensity_max = max(float(torch.quantile(intensity_values, 0.995)), 1.0e-12)
        error_max = max(float(torch.quantile(error.flatten(), 0.995)), 1.0e-12)
        z_value = total_distance * step / n_intervals
        relative = float(
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target).clamp_min(1.0e-12)
        )

        axes[0, column].imshow(target, cmap="inferno", vmin=0.0, vmax=intensity_max)
        axes[1, column].imshow(prediction, cmap="inferno", vmin=0.0, vmax=intensity_max)
        axes[2, column].imshow(error, cmap="magma", vmin=0.0, vmax=error_max)
        axes[0, column].set_title(f"z={z_value:g} m")
        axes[2, column].set_xlabel(f"relative L2={relative:.3e}")

    for row, label in enumerate(("target", "prediction", "absolute error")):
        axes[row, 0].set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Validation path {path_id}: best-checkpoint diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.blend_mse_weight < 0.0 or args.grad_clip < 0.0:
        raise ValueError("Loss weight and gradient clipping threshold cannot be negative")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)

    logger.info("Scanning chunk metadata in %s", args.data_dir)
    chunks = scan_turpy_chunks(args.data_dir, args.chunk_pattern)
    path_splits = split_path_ids(
        chunks,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    mode_splits = {
        name: mode_combination_ids_for_paths(chunks, path_ids)
        for name, path_ids in path_splits.items()
    }
    logger.info("Computing training-only normalization (one chunk in memory at a time)")
    normalization = compute_chunked_normalization(
        chunks,
        path_splits["train"],
        scan_batch_size=args.normalization_batch_size,
    )

    train_set = ChunkedTurpyDataset(chunks, path_splits["train"], normalization)
    val_set = ChunkedTurpyDataset(chunks, path_splits["val"], normalization)
    test_set = ChunkedTurpyDataset(chunks, path_splits["test"], normalization)
    train_sampler = ChunkShuffleSampler(train_set, seed=args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    height, width, input_channels = chunks[0].sample_shape
    model_kwargs = {
        "input_channels": input_channels,
        "modes_y": args.modes_y,
        "modes_x": args.modes_x,
        "width": args.width,
        "layers": args.layers,
        "use_skip": args.use_skip,
    }
    model = FNO2d(**model_kwargs).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1.0e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1.0e-6
        )

    split_manifest = {
        "split_unit": args.split_unit,
        "seed": args.seed,
        "chunk_files": [info.path.name for info in chunks],
        "train_mode_combination_ids": mode_splits["train"],
        "validation_mode_combination_ids": mode_splits["val"],
        "test_mode_combination_ids": mode_splits["test"],
        "train_path_ids": path_splits["train"],
        "validation_path_ids": path_splits["val"],
        "test_path_ids": path_splits["test"],
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "test_samples": len(test_set),
    }
    save_json(args.output_dir / "split_manifest.json", split_manifest)
    save_json(
        args.output_dir / "config.json",
        {**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Data: %d chunks, grid=%dx%d, input=%d channels, distance=%g m",
        len(chunks), height, width, input_channels, chunks[0].total_distance,
    )
    logger.info(
        "Paths train/val/test: %d/%d/%d; samples: %d/%d/%d",
        len(path_splits["train"]), len(path_splits["val"]), len(path_splits["test"]),
        len(train_set), len(val_set), len(test_set),
    )
    logger.info(
        "Mode combinations train/val/test: %d/%d/%d; split-unit=%s",
        len(mode_splits["train"]), len(mode_splits["val"]),
        len(mode_splits["test"]), args.split_unit,
    )
    logger.info("Normalization: %s", normalization.state_dict())
    logger.info("Device: %s; model parameters: %d", device, parameter_count)
    logger.info(
        "Early stopping: patience=%d, minimum improvement=%g",
        args.early_stopping_patience,
        args.early_stopping_min_delta,
    )
    if args.workers > 0:
        logger.info("Each worker may cache one full chunk; use --workers 0 for minimum RAM")

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_path = args.output_dir / "best.pt"
    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = objective(
                prediction, target, args.loss, args.blend_mse_weight, normalization
            )
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_total += loss.item() * x.shape[0]
            train_count += x.shape[0]

        train_loss = train_total / train_count
        train_set.clear_cache()
        validation = evaluate(
            model,
            val_loader,
            device,
            args.loss,
            args.blend_mse_weight,
            normalization,
            chunks[0].total_distance,
        )
        val_set.clear_cache()
        learning_rate = optimizer.param_groups[0]["lr"]
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(validation["loss"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_normalized_mse": validation["normalized_mse"],
            "val_physical_mse": validation["physical_mse"],
            "val_relative_l2": validation["relative_l2"],
            "val_rho0_baseline_physical_mse": validation[
                "rho0_baseline_physical_mse"
            ],
            "val_rho0_baseline_relative_l2": validation[
                "rho0_baseline_relative_l2"
            ],
            "learning_rate": learning_rate,
        }
        history.append(record)
        save_json(args.output_dir / "history.json", history)
        logger.info(
            (
                "epoch %04d | train=%.6e | val=%.6e | val_rel=%.6e "
                "| rho0_rel=%.6e | lr=%.3e"
            ),
            epoch,
            train_loss,
            validation["loss"],
            validation["relative_l2"],
            validation["rho0_baseline_relative_l2"],
            learning_rate,
        )
        if validation["loss"] < best_loss - args.early_stopping_min_delta:
            best_loss = validation["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "normalization": normalization.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "split_unit": args.split_unit,
                    "input_schema": "rho0 + delta_n-or-zero slots + history_fraction",
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                early_stopped = True
                logger.info(
                    (
                        "Early stopping at epoch %04d: validation loss did not "
                        "improve for %d epochs (best=%.6e)"
                    ),
                    epoch,
                    args.early_stopping_patience,
                    best_loss,
                )
                break

    train_set.clear_cache()
    val_set.clear_cache()
    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_json(
        args.output_dir / "validation_metrics.json",
        checkpoint["validation_metrics"],
    )
    save_prediction_diagnostics(
        model,
        val_set,
        path_splits["val"][0],
        normalization,
        chunks[0].total_distance,
        device,
        args.output_dir / "validation_predictions.png",
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        args.loss,
        args.blend_mse_weight,
        normalization,
        chunks[0].total_distance,
    )
    test_set.clear_cache()
    test_metrics["best_epoch"] = checkpoint["epoch"]
    test_metrics["epochs_completed"] = history[-1]["epoch"]
    test_metrics["early_stopped"] = early_stopped
    test_metrics["early_stopping_patience"] = args.early_stopping_patience
    test_metrics["split_unit"] = args.split_unit
    test_metrics["train_mode_combinations"] = len(mode_splits["train"])
    test_metrics["validation_mode_combinations"] = len(mode_splits["val"])
    test_metrics["test_mode_combinations"] = len(mode_splits["test"])
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    logger.info(
        (
            "test at epoch %04d | loss=%.6e | relative_l2=%.6e "
            "| rho0_relative_l2=%.6e | physical_mse=%.6e"
        ),
        checkpoint["epoch"],
        test_metrics["loss"],
        test_metrics["relative_l2"],
        test_metrics["rho0_baseline_relative_l2"],
        test_metrics["physical_mse"],
    )
    logger.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
