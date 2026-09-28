"""Compare direct and recursively restarted predictions from a TurPy FNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utilities import FNO2d, Normalization, image_comparison_metrics, load_turpy_file


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
    parser.add_argument(
        "--full-rollout",
        action="store_true",
        help="Restart from each predicted intensity through the final z step.",
    )
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


def load_path_examples(
    data_dir: Path, pattern: str, path_id: int
) -> tuple[torch.Tensor, torch.Tensor, int, float, Path]:
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
        order = torch.argsort(steps)
        expected_steps = torch.arange(1, n_intervals + 1)
        if not torch.equal(steps[order], expected_steps):
            raise ValueError(
                f"Expected exactly one example at each of {n_intervals} steps "
                f"for path {path_id} in {path}"
            )
        ordered_indices = indices[order]
        x = chunk["X"].index_select(0, ordered_indices).float().clone()
        y = chunk["Y"].index_select(0, ordered_indices).float().clone()
        total_distance = float(chunk["total_distance"])
        del chunk
        if x.shape[-1] != n_intervals + 2:
            raise ValueError("Saved input channels do not match the z-grid")
        for index in range(1, n_intervals):
            if not torch.equal(x[index, ..., 0], x[0, ..., 0]):
                raise ValueError(f"Initial intensity changes within path {path_id}")
            if not torch.equal(
                x[index, ..., 1:index + 1], x[index - 1, ..., 1:index + 1]
            ):
                raise ValueError(f"Saved turbulence history changes within path {path_id}")
        return x, y, n_intervals, total_distance, path
    raise ValueError(f"Path ID {path_id} was not found in {data_dir}")


def load_first_two_examples(
    data_dir: Path, pattern: str, path_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, Path]:
    x, y, n_intervals, total_distance, path = load_path_examples(
        data_dir, pattern, path_id
    )
    return x[0], y[0], x[1], y[1], n_intervals, total_distance, path


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
    quality: dict[str, dict[str, float]],
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
        "True rho1",
        f"Predicted rho1\nSSIM={quality['rho1']['ssim']:.3f}  PSNR={quality['rho1']['psnr_db']:.2f} dB",
        "True rho2",
        f"Direct rho2\nSSIM={quality['direct']['ssim']:.3f}  PSNR={quality['direct']['psnr_db']:.2f} dB",
        f"True-rho1 restart\nSSIM={quality['teacher']['ssim']:.3f}  PSNR={quality['teacher']['psnr_db']:.2f} dB",
        f"Recursive rho2\nSSIM={quality['recursive']['ssim']:.3f}  PSNR={quality['recursive']['psnr_db']:.2f} dB",
        f"|recursive - direct|\nrel L2={quality['recursive_vs_direct']['relative_l2']:.3e}",
        f"|recursive - true|\nrel L2={quality['recursive']['relative_l2']:.3e}",
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
        axis.set_title(titles[index], fontsize=10)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_full_rollout_plot(
    path: Path,
    rho_z_true: torch.Tensor,
    rho_z_direct: torch.Tensor,
    rho_z_recursive: torch.Tensor,
    per_step: list[dict],
) -> None:
    final = per_step[-1]
    images = [
        rho_z_true,
        rho_z_direct,
        rho_z_recursive,
        (rho_z_direct - rho_z_true).abs(),
        (rho_z_recursive - rho_z_true).abs(),
        (rho_z_recursive - rho_z_direct).abs(),
    ]
    images = [image.squeeze().detach().cpu() for image in images]
    intensity_max = max(
        float(torch.quantile(torch.cat([image.flatten() for image in images[:3]]), 0.99)),
        1e-12,
    )
    titles = [
        "True rhoZ",
        f"Direct rhoZ\nSSIM={final['direct']['ssim']:.3f}  PSNR={final['direct']['psnr_db']:.2f} dB",
        f"Recursive rhoZ\nSSIM={final['recursive']['ssim']:.3f}  PSNR={final['recursive']['psnr_db']:.2f} dB",
        f"|direct - true|\nrel L2={final['direct']['relative_l2']:.3e}",
        f"|recursive - true|\nrel L2={final['recursive']['relative_l2']:.3e}",
        f"|recursive - direct|\nrel L2={final['recursive_vs_direct_relative_l2']:.3e}",
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    image_axes = [axes[0, 0], axes[0, 1], axes[0, 2],
                  axes[1, 0], axes[1, 1], axes[1, 2]]
    for index, axis in enumerate(image_axes):
        vmax = (
            intensity_max if index < 3 else
            max(float(torch.quantile(images[index].flatten(), 0.99)), 1e-12)
        )
        image = axis.imshow(images[index], cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(titles[index], fontsize=10)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.8)

    distances = [entry["z_m"] for entry in per_step]
    for label, key in (("Direct", "direct"), ("Recursive", "recursive")):
        axes[0, 3].plot(
            distances,
            [max(entry[key]["relative_l2"], 1e-12) for entry in per_step],
            label=label,
        )
        axes[1, 3].plot(
            distances, [entry[key]["ssim"] for entry in per_step], label=label
        )
    axes[0, 3].set_yscale("log")
    axes[0, 3].set_title("Error along propagation")
    axes[0, 3].set_ylabel("Relative L2")
    axes[1, 3].set_title("Structure along propagation")
    axes[1, 3].set_ylabel("SSIM")
    for axis in (axes[0, 3], axes[1, 3]):
        axis.set_xlabel("Distance z (m)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_full_rollout(
    args: argparse.Namespace,
    path_id: int,
    model: FNO2d,
    normalization: Normalization,
    device: torch.device,
) -> None:
    x, y, n_intervals, total_distance, chunk_path = load_path_examples(
        args.data_dir, args.chunk_pattern, path_id
    )
    if x.shape[-1] != model.input_channels:
        raise ValueError("Checkpoint input channel count does not match the saved examples")

    per_step: list[dict] = []
    direct_fields: list[torch.Tensor] = []
    recursive_fields: list[torch.Tensor] = []
    recursive_previous = None
    for index in range(n_intervals):
        saved_input = x[index].unsqueeze(0).to(device)
        truth = y[index].unsqueeze(0).to(device)
        direct = normalization.denormalize_target(
            model(normalized_saved_input(saved_input, normalization))
        )
        if index == 0:
            recursive = direct
        else:
            next_screen = saved_input[..., index + 1]
            recursive = normalization.denormalize_target(
                model(restarted_input(
                    recursive_previous, next_screen, n_intervals, normalization
                ))
            )
        if not torch.isfinite(direct).all() or not torch.isfinite(recursive).all():
            raise RuntimeError(
                f"Non-finite FNO prediction at step {index + 1}; "
                "the recursive rollout may have diverged"
            )
        step = index + 1
        per_step.append({
            "step": step,
            "z_m": total_distance * step / n_intervals,
            "direct": image_comparison_metrics(direct, truth),
            "recursive": image_comparison_metrics(recursive, truth),
            "recursive_vs_direct_relative_l2": relative_l2(recursive, direct),
            "recursive_negative_pixel_fraction": float((recursive < 0).float().mean()),
        })
        direct_fields.append(direct[0].cpu())
        recursive_fields.append(recursive[0].cpu())
        recursive_previous = recursive

    output_dir = args.output_dir or args.checkpoint.parent / f"rollout_comparison_path_{path_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(args.checkpoint),
        "chunk": str(chunk_path),
        "path_id": path_id,
        "n_intervals": n_intervals,
        "total_distance_m": total_distance,
        "final": per_step[-1],
        "per_step": per_step,
    }
    (output_dir / "full_rollout_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "rho0_true": x[0, ..., :1],
            "rho_true": y,
            "rho_direct": torch.stack(direct_fields),
            "rho_recursive": torch.stack(recursive_fields),
        },
        output_dir / "full_rollout_fields.pt",
    )
    save_full_rollout_plot(
        output_dir / "full_rollout_comparison.png",
        y[-1], direct_fields[-1], recursive_fields[-1], per_step,
    )
    final = per_step[-1]
    print(
        f"Path {path_id}: {n_intervals} steps to z={total_distance:g} m; "
        f"chunk={chunk_path}"
    )
    print(
        "Final relative L2: "
        f"direct={final['direct']['relative_l2']:.4e}, "
        f"recursive={final['recursive']['relative_l2']:.4e}, "
        f"recursive-vs-direct={final['recursive_vs_direct_relative_l2']:.4e}"
    )
    print(f"Saved full rollout to {output_dir}")


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
    if args.full_rollout:
        run_full_rollout(args, path_id, model, normalization, device)
        return
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
    quality = {
        "rho1": image_comparison_metrics(rho1_pred, rho1_true),
        "direct": image_comparison_metrics(direct, rho2_true),
        "recursive": image_comparison_metrics(recursive, rho2_true),
        "teacher": image_comparison_metrics(teacher, rho2_true),
        "recursive_vs_direct": image_comparison_metrics(recursive, direct),
    }

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
        "image_quality": quality,
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
        rho2_true, direct, recursive, teacher, quality,
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
