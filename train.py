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


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Normalization,
) -> torch.Tensor:
    prediction = normalization.denormalize_target(prediction)
    target = normalization.denormalize_target(target)
    error = (prediction - target).flatten(1).norm(dim=1)
    target_norm = target.flatten(1).norm(dim=1).clamp_min(1.0e-12)
    return (error / target_norm).mean()


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
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "normalized_mse": 0.0,
        "physical_mse": 0.0,
        "relative_l2": 0.0,
    }
    sample_count = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        prediction = model(x)
        batch_size = x.shape[0]
        normalized_mse = torch.mean((prediction - target).square())
        prediction_physical = normalization.denormalize_target(prediction)
        target_physical = normalization.denormalize_target(target)
        physical_mse = torch.mean((prediction_physical - target_physical).square())
        relative = relative_l2(prediction, target, normalization)
        loss = objective(prediction, target, loss_type, blend_mse_weight, normalization)
        totals["loss"] += loss.item() * batch_size
        totals["normalized_mse"] += normalized_mse.item() * batch_size
        totals["physical_mse"] += physical_mse.item() * batch_size
        totals["relative_l2"] += relative.item() * batch_size
        sample_count += batch_size
    return {name: value / sample_count for name, value in totals.items()}


def save_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.blend_mse_weight < 0.0 or args.grad_clip < 0.0:
        raise ValueError("Loss weight and gradient clipping threshold cannot be negative")

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
    if args.workers > 0:
        logger.info("Each worker may cache one full chunk; use --workers 0 for minimum RAM")

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_path = args.output_dir / "best.pt"
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
            model, val_loader, device, args.loss, args.blend_mse_weight, normalization
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
            "learning_rate": learning_rate,
        }
        history.append(record)
        save_json(args.output_dir / "history.json", history)
        logger.info(
            "epoch %04d | train=%.6e | val=%.6e | val_rel=%.6e | lr=%.3e",
            epoch, train_loss, validation["loss"], validation["relative_l2"], learning_rate,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
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

    train_set.clear_cache()
    val_set.clear_cache()
    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model, test_loader, device, args.loss, args.blend_mse_weight, normalization
    )
    test_set.clear_cache()
    test_metrics["best_epoch"] = checkpoint["epoch"]
    test_metrics["split_unit"] = args.split_unit
    test_metrics["train_mode_combinations"] = len(mode_splits["train"])
    test_metrics["validation_mode_combinations"] = len(mode_splits["val"])
    test_metrics["test_mode_combinations"] = len(mode_splits["test"])
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    logger.info(
        "test at epoch %04d | loss=%.6e | relative_l2=%.6e | physical_mse=%.6e",
        checkpoint["epoch"], test_metrics["loss"], test_metrics["relative_l2"],
        test_metrics["physical_mse"],
    )
    logger.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
