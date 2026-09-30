"""Train an FNO to predict the next intensity from a recent TurPy history window."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from train import objective, select_device, setup_logger
from utilities import (
    ChunkInfo,
    ChunkShuffleSampler,
    FNO2d,
    Normalization,
    load_turpy_file,
    mode_combination_ids_for_paths,
    scan_turpy_chunks,
    split_path_ids,
)


@dataclass(frozen=True)
class PathRows:
    chunk_index: int
    path_id: int
    rows: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FNO from rolling TurPy intensity and delta-n windows."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/turpy_chunks_4km"))
    parser.add_argument("--chunk-pattern", default="chunk_*.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/turpy_window_fno"))
    parser.add_argument("--window", type=int, default=10, help="Number of intensity/screen pairs in each input.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--learning-rate-min", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes-y", type=int, default=16)
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--no-skip", dest="use_skip", action="store_false", default=True)
    parser.add_argument("--loss", choices=("mse", "relative-l2", "blend"), default="relative-l2")
    parser.add_argument("--blend-mse-weight", type=float, default=0.1)
    parser.add_argument("--scheduler", choices=("cosine", "plateau"), default="plateau")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-unit", choices=("mode-combination", "path"), default="path")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def index_path_rows(chunks: list[ChunkInfo], window: int) -> dict[int, PathRows]:
    """Map each global path ID to sample rows ordered by propagation interval."""
    path_rows: dict[int, PathRows] = {}
    for chunk_index, info in enumerate(chunks):
        chunk = load_turpy_file(info.path)
        path_ids = chunk["path_ids"].long()
        n_intervals = info.n_z - 1
        for path_id in dict.fromkeys(info.path_ids):
            indices = torch.where(path_ids == path_id)[0]
            steps = torch.round(chunk["X"][indices, 0, 0, -1] * n_intervals).long()
            order = torch.argsort(steps)
            ordered_steps = steps[order]
            expected = torch.arange(1, n_intervals + 1)
            if indices.numel() != n_intervals or not torch.equal(ordered_steps, expected):
                raise ValueError(
                    f"Path {path_id} in {info.path} must contain exactly one sample "
                    f"for each of its {n_intervals} propagation intervals."
                )
            if n_intervals < window:
                raise ValueError(
                    f"Path {path_id} has {n_intervals} intervals, fewer than window={window}."
                )
            path_rows[int(path_id)] = PathRows(
                chunk_index=chunk_index,
                path_id=int(path_id),
                rows=tuple(int(value) for value in indices[order].tolist()),
            )
        del chunk
    return path_rows


def compute_window_normalization(
    chunks: list[ChunkInfo],
    paths: dict[int, PathRows],
    selected_path_ids: list[int],
) -> Normalization:
    """Compute intensity moments and screen RMS using training paths only."""
    selected = set(selected_path_ids)
    intensity_sum = intensity_sq_sum = delta_sq_sum = 0.0
    intensity_count = delta_count = 0
    for chunk_index, info in enumerate(chunks):
        chosen = [paths[path_id] for path_id in selected if paths[path_id].chunk_index == chunk_index]
        if not chosen:
            continue
        chunk = load_turpy_file(info.path)
        for path in chosen:
            rows = torch.tensor(path.rows, dtype=torch.long)
            x = chunk["X"].index_select(0, rows).float()
            y = chunk["Y"].index_select(0, rows).float()[..., 0]
            state_images = torch.cat((x[:1, ..., 0], y), dim=0)
            intensity_sum += state_images.sum(dtype=torch.float64).item()
            intensity_sq_sum += state_images.square().sum(dtype=torch.float64).item()
            intensity_count += state_images.numel()
            for interval in range(info.n_z - 1):
                screen = x[interval, ..., 1 + interval]
                delta_sq_sum += screen.square().sum(dtype=torch.float64).item()
                delta_count += screen.numel()
        del chunk
    if intensity_count == 0 or delta_count == 0:
        raise ValueError("Training paths contain no usable intensity or screen values")
    mean = intensity_sum / intensity_count
    variance = max(intensity_sq_sum / intensity_count - mean * mean, 1.0e-20)
    return Normalization(
        intensity_mean=mean,
        intensity_std=variance**0.5,
        delta_n_rms=max((delta_sq_sum / delta_count) ** 0.5, 1.0e-20),
    )


class WindowedTurpyDataset(Dataset):
    """Lazy rolling windows reconstructed from existing one-step TurPy chunks."""

    def __init__(
        self,
        chunks: list[ChunkInfo],
        paths: dict[int, PathRows],
        selected_path_ids: list[int],
        window: int,
        normalization: Normalization,
    ) -> None:
        self.chunks = chunks
        self.paths: list[PathRows] = []
        self.windows: list[tuple[int, int]] = []
        self.chunk_ranges: list[tuple[int, int]] = []
        self.window = window
        self.normalization = normalization
        self._cached_chunk_index = -1
        self._cached_chunk: dict | None = None
        selected_paths = [paths[path_id] for path_id in selected_path_ids]
        selected_paths.sort(key=lambda path: (path.chunk_index, path.path_id))
        for chunk_index in sorted({path.chunk_index for path in selected_paths}):
            range_start = len(self.windows)
            for path in (p for p in selected_paths if p.chunk_index == chunk_index):
                path_index = len(self.paths)
                self.paths.append(path)
                count = len(path.rows) - window + 1
                self.windows.extend((path_index, start) for start in range(count))
            self.chunk_ranges.append((range_start, len(self.windows)))
        if not self.windows:
            raise ValueError("No rolling windows were constructed")

    def __len__(self) -> int:
        return len(self.windows)

    def clear_cache(self) -> None:
        self._cached_chunk = None
        self._cached_chunk_index = -1

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cached_chunk"] = None
        state["_cached_chunk_index"] = -1
        return state

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path_index, start = self.windows[index]
        path = self.paths[path_index]
        if path.chunk_index != self._cached_chunk_index:
            self.clear_cache()
            self._cached_chunk = load_turpy_file(self.chunks[path.chunk_index].path)
            self._cached_chunk_index = path.chunk_index
        assert self._cached_chunk is not None
        chunk = self._cached_chunk
        rows = path.rows

        intensities = []
        screens = []
        for state_index in range(start, start + self.window):
            if state_index == 0:
                state = chunk["X"][rows[0], ..., 0]
            else:
                state = chunk["Y"][rows[state_index - 1], ..., 0]
            intensities.append(state.float())
            screens.append(
                chunk["X"][rows[state_index], ..., 1 + state_index].float()
            )
        intensity_input = torch.stack(intensities, dim=-1)
        delta_input = torch.stack(screens, dim=-1)
        x = torch.cat((intensity_input, delta_input), dim=-1)
        x[..., :self.window] = (
            x[..., :self.window] - self.normalization.intensity_mean
        ) / self.normalization.intensity_std
        x[..., self.window:] /= self.normalization.delta_n_rms

        target_row = rows[start + self.window - 1]
        y = chunk["Y"][target_row].float()
        y = self.normalization.normalize_target(y)
        return {"x": x, "y": y, "path_id": torch.tensor(path.path_id)}


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    blend_mse_weight: float,
    normalization: Normalization,
    window: int,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "normalized_mse": 0.0, "physical_mse": 0.0,
              "relative_l2": 0.0, "last_intensity_baseline_relative_l2": 0.0}
    count = 0
    for batch in loader:
        x, target = batch["x"].to(device), batch["y"].to(device)
        prediction = model(x)
        normalized_mse = (prediction - target).square().flatten(1).mean(dim=1)
        prediction_physical = normalization.denormalize_target(prediction)
        target_physical = normalization.denormalize_target(target)
        relative = (prediction_physical - target_physical).flatten(1).norm(dim=1)
        relative /= target_physical.flatten(1).norm(dim=1).clamp_min(1.0e-12)
        last_intensity = x[..., window - 1].unsqueeze(-1)
        last_physical = normalization.denormalize_target(last_intensity)
        baseline_relative = (last_physical - target_physical).flatten(1).norm(dim=1)
        baseline_relative /= target_physical.flatten(1).norm(dim=1).clamp_min(1.0e-12)
        if loss_type == "mse":
            losses = normalized_mse
        elif loss_type == "relative-l2":
            losses = relative
        else:
            losses = relative + blend_mse_weight * normalized_mse
        n = x.shape[0]
        totals["loss"] += losses.sum().item()
        totals["normalized_mse"] += normalized_mse.sum().item()
        totals["physical_mse"] += (prediction_physical - target_physical).square().flatten(1).mean(1).sum().item()
        totals["relative_l2"] += relative.sum().item()
        totals["last_intensity_baseline_relative_l2"] += baseline_relative.sum().item()
        count += n
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.window < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--window, --epochs, and --batch-size must be positive")
    if args.blend_mse_weight < 0 or args.grad_clip < 0:
        raise ValueError("Loss weight and gradient clipping threshold cannot be negative")
    if args.early_stopping_patience < 1 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping settings are invalid")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    logger = setup_logger(args.output_dir)

    logger.info("Scanning chunks in %s", args.data_dir)
    chunks = scan_turpy_chunks(args.data_dir, args.chunk_pattern)
    paths = index_path_rows(chunks, args.window)
    path_splits = split_path_ids(
        chunks, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        seed=args.seed, split_unit=args.split_unit,
    )
    mode_splits = {
        name: mode_combination_ids_for_paths(chunks, ids)
        for name, ids in path_splits.items()
    }
    normalization = compute_window_normalization(chunks, paths, path_splits["train"])
    train_set = WindowedTurpyDataset(chunks, paths, path_splits["train"], args.window, normalization)
    val_set = WindowedTurpyDataset(chunks, paths, path_splits["val"], args.window, normalization)
    test_set = WindowedTurpyDataset(chunks, paths, path_splits["test"], args.window, normalization)
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

    input_channels = 2 * args.window
    model_kwargs = {
        "input_channels": input_channels,
        "modes_y": args.modes_y,
        "modes_x": args.modes_x,
        "width": args.width,
        "layers": args.layers,
        "use_skip": args.use_skip,
    }
    model = FNO2d(**model_kwargs).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1.0e-6)

    windows_per_path = chunks[0].n_z - 1 - args.window + 1
    manifest = {
        "input_schema": "rho_window_then_delta_n_window",
        "window_length": args.window,
        "windows_per_path": windows_per_path,
        "split_unit": args.split_unit,
        "seed": args.seed,
        "chunk_files": [info.path.name for info in chunks],
        "train_path_ids": path_splits["train"],
        "validation_path_ids": path_splits["val"],
        "test_path_ids": path_splits["test"],
        "train_mode_combination_ids": mode_splits["train"],
        "validation_mode_combination_ids": mode_splits["val"],
        "test_mode_combination_ids": mode_splits["test"],
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "test_samples": len(test_set),
    }
    save_json(args.output_dir / "split_manifest.json", manifest)
    save_json(args.output_dir / "config.json", {
        **vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir),
    })
    logger.info("Input channels: %d (%d intensity + %d delta_n)", input_channels, args.window, args.window)
    logger.info("Windows per path: %d; samples train/val/test: %d/%d/%d",
                windows_per_path, len(train_set), len(val_set), len(test_set))
    logger.info("Normalization: %s; device=%s", normalization.state_dict(), device)

    history: list[dict] = []
    best_loss = float("inf")
    best_path = args.output_dir / "best.pt"
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            x, target = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = objective(prediction, target, args.loss, args.blend_mse_weight, normalization)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        train_set.clear_cache()
        validation = evaluate(model, val_loader, device, args.loss, args.blend_mse_weight, normalization, args.window)
        val_set.clear_cache()
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(validation["loss"])
        record = {"epoch": epoch, "train_loss": total / count, **{f"val_{k}": v for k, v in validation.items()}}
        history.append(record)
        save_json(args.output_dir / "history.json", history)
        logger.info("epoch %04d | train=%.6e | val=%.6e | val_rel=%.6e",
                    epoch, record["train_loss"], validation["loss"], validation["relative_l2"])
        if validation["loss"] < best_loss - args.early_stopping_min_delta:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "normalization": normalization.state_dict(),
                "window_length": args.window,
                "input_schema": "rho_window_then_delta_n_window",
                "epoch": epoch,
                "validation_metrics": validation,
                "split_unit": args.split_unit,
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    train_set.clear_cache()
    val_set.clear_cache()
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, args.loss, args.blend_mse_weight, normalization, args.window)
    test_set.clear_cache()
    test_metrics.update({"best_epoch": checkpoint["epoch"], "window_length": args.window})
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    save_json(args.output_dir / "validation_metrics.json", checkpoint["validation_metrics"])
    logger.info("Test | loss=%.6e | relative_l2=%.6e | last-frame baseline rel=%.6e",
                test_metrics["loss"], test_metrics["relative_l2"],
                test_metrics["last_intensity_baseline_relative_l2"])
    logger.info("Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
