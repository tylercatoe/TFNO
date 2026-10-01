"""Streaming test-path selection and compact, resumable inversion metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utilities import load_turpy_file


def test_path_ids(manifest_path: Path, max_paths: int | None = None) -> list[int]:
    """Select held-out paths, not the individual z-step training examples."""
    if max_paths is not None and max_paths < 1:
        raise ValueError("--max-paths must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [int(value) for value in manifest["test_path_ids"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("The manifest must contain distinct test path IDs")
    return ids[:max_paths]


def iter_final_test_examples(
    data_dir: Path, pattern: str, selected_path_ids: Iterable[int]
):
    """Read one chunk at a time and yield one final-z sample per requested path."""
    selected = set(selected_path_ids)
    if not selected:
        raise ValueError("No test paths were selected")
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No chunks matching {pattern!r} in {data_dir}")
    seen: set[int] = set()
    for path in paths:
        chunk = load_turpy_file(path)
        n_intervals = int(chunk["n_z"]) - 1
        x_all, y_all, path_ids = chunk["X"], chunk["Y"], chunk["path_ids"]
        steps = torch.round(x_all[:, 0, 0, -1] * n_intervals).long()
        path_metadata = chunk.get("path_metadata") or []
        local_ids = list(dict.fromkeys(int(value) for value in path_ids.tolist()))
        mode_by_id = {
            path_id: info.get("mode_combination_index")
            for path_id, info in zip(local_ids, path_metadata)
        }
        final_indices = torch.nonzero(steps == n_intervals, as_tuple=True)[0].tolist()
        for index in final_indices:
            path_id = int(path_ids[index])
            if path_id not in selected:
                continue
            if path_id in seen:
                raise ValueError(f"Multiple final samples found for test path {path_id}")
            seen.add(path_id)
            x = x_all[index].float().clone()
            y = y_all[index, ..., 0].float().clone()
            if x.shape[-1] != n_intervals + 2:
                raise ValueError(f"Input channels disagree with n_z in {path}")
            metadata = {
                "n_intervals": n_intervals,
                "total_distance": float(chunk["total_distance"]),
                "wavelength": float(chunk["wavelength"]),
                "n0": float(chunk["n0"]),
                "dx": float(chunk["dx"]),
                "initial_phase": chunk.get("initial_phase"),
                "mode_combination_index": mode_by_id.get(path_id),
            }
            yield path_id, x, y, metadata, path
        del chunk
    missing = selected - seen
    if missing:
        raise ValueError(f"Missing final examples for {len(missing)} test paths: {sorted(missing)[:10]}")


def numeric_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    numbers = list(values)
    finite = sorted(float(value) for value in numbers if value is not None and math.isfinite(float(value)))
    if not finite:
        return {"count": len(numbers), "finite_count": 0, "mean": None,
                "median": None, "min": None, "max": None, "p10": None,
                "p25": None, "p75": None, "p90": None}

    def quantile(fraction: float) -> float:
        position = (len(finite) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(finite) - 1)
        weight = position - low
        return finite[low] * (1 - weight) + finite[high] * weight

    return {"count": len(numbers), "finite_count": len(finite),
            "mean": mean(finite), "median": median(finite),
            "min": finite[0], "max": finite[-1],
            "p10": quantile(0.10), "p25": quantile(0.25),
            "p75": quantile(0.75), "p90": quantile(0.90)}


def begin_test_set_run(output_dir: Path, config: dict, resume: bool) -> tuple[list[dict], Path]:
    """Keep only metric rows per path; a resumed run must use identical settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    rows_path = output_dir / "path_metrics.jsonl"
    if resume:
        if not config_path.is_file() or not rows_path.is_file():
            raise FileNotFoundError("--resume requires run_config.json and path_metrics.jsonl")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume settings differ from run_config.json; use a new --output-dir")
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
        if len({row["path_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate path IDs in existing path_metrics.jsonl")
        return rows, rows_path
    if config_path.exists() or rows_path.exists():
        raise FileExistsError(f"Test-set results already exist in {output_dir}; use --resume or a new --output-dir")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows_path.write_text("", encoding="utf-8")
    return [], rows_path


def append_path_metrics(path: Path, record: dict) -> None:
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        stream.flush()


def summarize_screen_records(records: list[dict], oracle_key: str) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty test set")
    n_screens = len(records[0]["screen_metrics_evaluation_only"])
    if any(len(row["screen_metrics_evaluation_only"]) != n_screens for row in records):
        raise ValueError("Test paths do not have matching screen counts")
    identifiable = range(n_screens - 1)
    per_screen = [
        {
            "screen": index,
            "correlation": numeric_summary(row["screen_metrics_evaluation_only"][index]["correlation"] for row in records),
            "relative_l2": numeric_summary(row["screen_metrics_evaluation_only"][index]["relative_l2"] for row in records),
        }
        for index in identifiable
    ]
    per_path_abs_correlation = [
        median(abs(row["screen_metrics_evaluation_only"][index]["correlation"])
               for index in identifiable)
        for row in records
    ] if n_screens > 1 else []
    return {
        "test_path_count": len(records),
        "n_screens": n_screens,
        "last_screen_excluded_as_unidentifiable": True,
        "final_image_relative_l2": numeric_summary(row["final_image_metrics"]["relative_l2"] for row in records),
        "final_image_ssim": numeric_summary(row["final_image_metrics"]["ssim"] for row in records),
        "final_image_psnr_db": numeric_summary(row["final_image_metrics"]["psnr_db"] for row in records),
        "oracle_final_image_relative_l2_evaluation_only": numeric_summary(
            row[oracle_key]["relative_l2"] for row in records
        ),
        "per_path_median_absolute_screen_correlation": numeric_summary(per_path_abs_correlation),
        "per_screen_evaluation_only": per_screen,
    }


def save_screen_test_set_plot(path: Path, records: list[dict], summary: dict, title: str,
                              include_correlations: bool = True) -> None:
    figure, axes = plt.subplots(1, 3 if include_correlations else 1,
                               figsize=(14 if include_correlations else 6, 4),
                               constrained_layout=True)
    final_errors = [row["final_image_metrics"]["relative_l2"] for row in records]
    final_axis = axes[0] if include_correlations else axes
    final_axis.hist(final_errors, bins=min(20, max(1, len(final_errors))), color="tab:blue")
    final_axis.set_xlabel("Final intensity relative L2")
    final_axis.set_ylabel("Test paths")
    if include_correlations:
        correlations = [
            median(abs(item["correlation"]) for item in row["screen_metrics_evaluation_only"][:-1])
            for row in records
        ]
        axes[1].hist(correlations, bins=min(20, max(1, len(correlations))), color="tab:orange")
        axes[1].set_xlabel("Median |screen correlation| per path")
        axes[1].set_ylabel("Test paths")
        per_screen = summary["per_screen_evaluation_only"]
        indices = [item["screen"] for item in per_screen]
        medians = [item["correlation"]["median"] for item in per_screen]
        lows = [item["correlation"]["p25"] for item in per_screen]
        highs = [item["correlation"]["p75"] for item in per_screen]
        axes[2].plot(indices, medians, marker="o", label="Median")
        axes[2].fill_between(indices, lows, highs, alpha=0.25, label="IQR")
        axes[2].axhline(0, color="0.5", linewidth=0.8)
        axes[2].set_ylim(-1.05, 1.05)
        axes[2].set_xlabel("Screen index (last excluded)")
        axes[2].set_ylabel("True/estimate correlation")
        axes[2].legend()
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarize_rho0_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty test set")
    return {
        "test_path_count": len(records),
        "rho0_relative_l2": numeric_summary(row["relative_initial_l2"] for row in records),
        "rho0_ssim": numeric_summary(row["image_quality"]["initial"]["ssim"] for row in records),
        "rho0_psnr_db": numeric_summary(row["image_quality"]["initial"]["psnr_db"] for row in records),
        "initial_guess_rho0_relative_l2": numeric_summary(row["relative_guess_l2"] for row in records),
        "final_image_relative_l2": numeric_summary(row["relative_final_l2"] for row in records),
        "final_image_ssim": numeric_summary(row["image_quality"]["final"]["ssim"] for row in records),
        "final_image_psnr_db": numeric_summary(row["image_quality"]["final"]["psnr_db"] for row in records),
        "oracle_fno_final_image_relative_l2_evaluation_only": numeric_summary(
            row["relative_true_input_forward_l2"] for row in records
        ),
    }


def save_rho0_test_set_plot(path: Path, records: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    n_bins = min(20, max(1, len(records)))
    axes[0].hist([row["relative_initial_l2"] for row in records], bins=n_bins,
                 color="tab:purple")
    axes[0].set_xlabel("Recovered rho(0) relative L2")
    axes[0].set_ylabel("Test paths")
    axes[1].hist([row["relative_final_l2"] for row in records], bins=n_bins,
                 color="tab:blue")
    axes[1].set_xlabel("Fitted rho(Z) relative L2")
    axes[1].set_ylabel("Test paths")
    axes[2].scatter([row["relative_true_input_forward_l2"] for row in records],
                    [row["relative_initial_l2"] for row in records], s=16, alpha=0.7)
    axes[2].set_xlabel("FNO rho(Z) error with true rho(0)")
    axes[2].set_ylabel("Recovered rho(0) relative L2")
    figure.suptitle("FNO rho(0) inversion: held-out paths")
    figure.savefig(path, dpi=160)
    plt.close(figure)
