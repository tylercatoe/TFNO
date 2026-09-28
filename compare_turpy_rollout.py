"""Compare direct and restarted two-step predictions from a TurPy FNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utilities import FNO2d, Normalization, load_turpy_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/turpy_fno_4km_ic_split/best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("turpy_chunks_4km"))
    parser.add_argument("--chunk-pattern", default="chunk_*.pt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--path-id", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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


def load_first_two_examples(
    data_dir: Path, pattern: str, path_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, Path]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No chunks matching {pattern!r} in {data_dir}")
    for path in paths:
        chunk = load_turpy_file(path)
        indices = torch.nonzero(chunk["path_ids"] == path_id, as_tuple=True)[0]
        if indices.numel() == 0:
            del chunk
            continue
        n_intervals = int(chunk["n_z"]) - 1
        if n_intervals < 2:
            raise ValueError("A two-step comparison requires at least two intervals")
        fractions = chunk["X"].index_select(0, indices)[:, 0, 0, -1]
        steps = torch.round(fractions * n_intervals).long()
        selected = []
        for step in (1, 2):
            matching = indices[steps == step]
            if matching.numel() != 1:
                raise ValueError(
                    f"Expected one step-{step} example for path {path_id} in {path}"
                )
            selected.append(int(matching[0]))
        first, second = selected
        x1 = chunk["X"][first].float().clone()
        y1 = chunk["Y"][first].float().clone()
        x2 = chunk["X"][second].float().clone()
        y2 = chunk["Y"][second].float().clone()
        total_distance = float(chunk["total_distance"])
        del chunk
        return x1, y1, x2, y2, n_intervals, total_distance, path
    raise ValueError(f"Path ID {path_id} was not found in {data_dir}")


def normalized_saved_input(value: torch.Tensor, normalization: Normalization) -> torch.Tensor:
    result = value.clone()
    result[..., 0] = normalization.normalize_target(result[..., 0])
    result[..., 1:-1] /= normalization.delta_n_rms
    return result


def restarted_input(
    rho1: torch.Tensor,
    delta_n1: torch.Tensor,
    n_intervals: int,
    normalization: Normalization,
) -> torch.Tensor:
    """Build [rho1, delta_n1, 0, ..., 0, 1/n_intervals] in training units."""
    batch, height, width, channels = rho1.shape
    if channels != 1 or delta_n1.shape != (batch, height, width):
        raise ValueError("rho1 must be [B,H,W,1] and delta_n1 must be [B,H,W]")
    result = rho1.new_zeros((batch, height, width, n_intervals + 2))
    result[..., :1] = normalization.normalize_target(rho1)
    result[..., 1] = delta_n1 / normalization.delta_n_rms
    result[..., -1] = 1.0 / n_intervals
    return result


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).norm() / target.norm().clamp_min(1e-12))


def physical_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).square().mean())


def save_comparison_plot(
    path: Path,
    rho1_true: torch.Tensor,
    rho1_pred: torch.Tensor,
    rho2_true: torch.Tensor,
    direct: torch.Tensor,
    recursive: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    images = [
        rho1_true, rho1_pred, rho2_true, direct,
        teacher, recursive, (recursive - direct).abs(), (recursive - rho2_true).abs(),
    ]
    images = [image.squeeze().detach().cpu() for image in images]
    rho1_max = max(float(torch.quantile(torch.cat((images[0], images[1])).flatten(), 0.99)), 1e-12)
    rho2_max = max(
        float(torch.quantile(torch.cat((images[2], images[3], images[4], images[5])).flatten(), 0.99)),
        1e-12,
    )
    titles = [
        "True rho1", "Predicted rho1", "True rho2", "Direct rho2",
        "True-rho1 restart", "Recursive rho2", "|recursive - direct|", "|recursive - true|",
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index < 2:
            vmax = rho1_max
        elif index < 6:
            vmax = rho2_max
        else:
            vmax = max(float(torch.quantile(images[index].flatten(), 0.99)), 1e-12)
        image = axis.imshow(images[index], cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(titles[index])
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
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

    path_id = choose_path_id(args)
    x1, y1, x2, y2, n_intervals, total_distance, chunk_path = load_first_two_examples(
        args.data_dir, args.chunk_pattern, path_id
    )
    if x1.shape[-1] != model.input_channels or x2.shape[-1] != model.input_channels:
        raise ValueError("Checkpoint input channel count does not match the saved examples")
    if not torch.equal(x1[..., 0], x2[..., 0]):
        raise ValueError("The two examples do not share the same initial intensity")
    if not torch.equal(x1[..., 1], x2[..., 1]):
        raise ValueError("The two examples do not share the same first turbulence screen")

    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device)
    rho1_true, rho2_true = y1.unsqueeze(0).to(device), y2.unsqueeze(0).to(device)
    rho1_pred = normalization.denormalize_target(
        model(normalized_saved_input(x1, normalization))
    )
    direct = normalization.denormalize_target(
        model(normalized_saved_input(x2, normalization))
    )
    delta_n1 = x2[..., 2]
    recursive = normalization.denormalize_target(
        model(restarted_input(rho1_pred, delta_n1, n_intervals, normalization))
    )
    teacher = normalization.denormalize_target(
        model(restarted_input(rho1_true, delta_n1, n_intervals, normalization))
    )

    output_dir = args.output_dir or args.checkpoint.parent / f"rollout_comparison_path_{path_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dz = total_distance / n_intervals
    metrics = {
        "checkpoint": str(args.checkpoint),
        "chunk": str(chunk_path),
        "path_id": path_id,
        "dz_m": dz,
        "z1_m": dz,
        "z2_m": 2 * dz,
        "rho1_relative_l2": relative_l2(rho1_pred, rho1_true),
        "direct_rho2_relative_l2": relative_l2(direct, rho2_true),
        "recursive_rho2_relative_l2": relative_l2(recursive, rho2_true),
        "true_rho1_restart_relative_l2": relative_l2(teacher, rho2_true),
        "recursive_vs_direct_relative_l2": relative_l2(recursive, direct),
        "direct_rho2_physical_mse": physical_mse(direct, rho2_true),
        "recursive_rho2_physical_mse": physical_mse(recursive, rho2_true),
        "true_rho1_restart_physical_mse": physical_mse(teacher, rho2_true),
        "rho1_negative_pixel_fraction": float((rho1_pred < 0).float().mean()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "rho1_true": rho1_true.cpu(),
            "rho1_predicted": rho1_pred.cpu(),
            "rho2_true": rho2_true.cpu(),
            "rho2_direct": direct.cpu(),
            "rho2_recursive": recursive.cpu(),
            "rho2_true_rho1_restart": teacher.cpu(),
        },
        output_dir / "fields.pt",
    )
    save_comparison_plot(
        output_dir / "comparison.png", rho1_true, rho1_pred,
        rho2_true, direct, recursive, teacher,
    )
    print(f"Path {path_id}: z1={dz:g} m, z2={2 * dz:g} m; chunk={chunk_path}")
    print(
        "Relative L2 at z2: "
        f"direct={metrics['direct_rho2_relative_l2']:.4e}, "
        f"recursive={metrics['recursive_rho2_relative_l2']:.4e}, "
        f"true-rho1 restart={metrics['true_rho1_restart_relative_l2']:.4e}"
    )
    print(f"Saved comparison to {output_dir}")


if __name__ == "__main__":
    main()
