"""Infer phase screens from rho(0) and rho(Z) through the frozen TurPy FNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from inversion_evaluation import (
    append_path_metrics, begin_test_set_run, iter_final_test_examples,
    save_screen_test_set_plot, summarize_screen_records, test_path_ids,
)
from utilities import FNO2d, Normalization, image_comparison_metrics, load_turpy_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/turpy_fno_4km_ic_split/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/turpy_chunks_4km"))
    parser.add_argument("--chunk-pattern", default="chunk_*.pt")
    parser.add_argument("--split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--path-id", type=int)
    parser.add_argument("--all-test-paths", action="store_true", help="Evaluate one final-z case per held-out test path.")
    parser.add_argument("--max-paths", type=int, help="Limit test paths for a smoke test.")
    parser.add_argument("--example-plots", type=int, default=3, help="Number of test paths with detailed plots/fields.")
    parser.add_argument("--resume", action="store_true", help="Continue a test-set run using existing metric rows.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-its", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--init-std", type=float, default=0.05, help="Initial normalized-screen standard deviation.")
    parser.add_argument("--regularization", choices=("none", "l2", "smooth"), default="smooth")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_path_id(manifest: Path, split: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; pass --path-id to select a path explicitly")
    selected = json.loads(manifest.read_text(encoding="utf-8"))[f"{split}_path_ids"]
    if not selected:
        raise ValueError(f"No paths in the {split} split")
    return int(selected[0])


def load_final_example(data_dir: Path, pattern: str, path_id: int) -> tuple[torch.Tensor, torch.Tensor, dict, Path]:
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
        steps = torch.round(chunk["X"][indices, 0, 0, -1] * n_intervals).long()
        final_indices = indices[steps == n_intervals]
        if final_indices.numel() != 1:
            raise ValueError(f"Expected one final-step sample for path {path_id} in {path}")
        index = int(final_indices[0])
        x = chunk["X"][index].float().clone()
        y = chunk["Y"][index, ..., 0].float().clone()
        metadata = {"n_intervals": n_intervals, "total_distance": float(chunk["total_distance"]),
                    "wavelength": float(chunk["wavelength"])}
        if x.shape[-1] != n_intervals + 2:
            raise ValueError("Saved input channels do not match the number of intervals")
        return x, y, metadata, path
    raise ValueError(f"Path ID {path_id} was not found in {data_dir}")


def screen_penalty(screens: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "l2":
        return screens.square().mean()
    if kind == "smooth":
        return ((screens[:, 1:, :] - screens[:, :-1, :]).square().mean()
                + (screens[:, :, 1:] - screens[:, :, :-1]).square().mean())
    return screens.new_zeros(())


def forward_model(model: FNO2d, normalization: Normalization, rho0: torch.Tensor,
                  screens_normalized: torch.Tensor) -> torch.Tensor:
    height, width = rho0.shape
    model_input = torch.cat((
        normalization.normalize_target(rho0)[None, ..., None],
        screens_normalized.permute(1, 2, 0)[None],
        torch.ones((1, height, width, 1), device=rho0.device, dtype=rho0.dtype),
    ), dim=-1)
    return normalization.denormalize_target(model(model_input))[0, ..., 0]


def screen_scores(estimated: torch.Tensor, true: torch.Tensor) -> list[dict]:
    scores = []
    for index, (estimate, reference) in enumerate(zip(estimated, true)):
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        denom = estimate.norm() * reference.norm()
        scores.append({
            "screen": index,
            "identifiable_from_rhoZ": index < true.shape[0] - 1,
            "relative_l2": float((estimate - reference).norm() / reference.norm().clamp_min(1e-12)),
            "correlation": float((estimate * reference).sum() / denom.clamp_min(1e-12)),
        })
    return scores


def save_plots(output_dir: Path, rho0: torch.Tensor, observed: torch.Tensor,
               predicted: torch.Tensor, true_phase: torch.Tensor, estimated_phase: torch.Tensor,
               history: list[dict], scores: list[dict], quality: dict) -> None:
    rho0, observed, predicted = [value.detach().cpu() for value in (rho0, observed, predicted)]
    true_phase, estimated_phase = [value.detach().cpu() for value in (true_phase, estimated_phase)]
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    images = [rho0, observed, predicted, (predicted - observed).abs()]
    titles = ["Known rho(0)", "Observed rho(Z)",
              f"FNO fit\nSSIM={quality['ssim']:.3f}, PSNR={quality['psnr_db']:.1f} dB",
              f"Absolute error\nrel L2={quality['relative_l2']:.3e}"]
    vmax = max(float(torch.quantile(torch.cat((observed.flatten(), predicted.flatten())), 0.99)), 1e-12)
    for index, axis in enumerate(axes):
        upper = vmax if index < 3 else max(float(torch.quantile(images[index].flatten(), 0.99)), 1e-12)
        im = axis.imshow(images[index], cmap="magma", vmin=0, vmax=upper)
        axis.set_title(titles[index])
        axis.set_axis_off()
        figure.colorbar(im, ax=axis, shrink=0.75)
    figure.savefig(output_dir / "final_intensity.png", dpi=160)
    plt.close(figure)

    selected = sorted(set([0, true_phase.shape[0] // 4, true_phase.shape[0] // 2,
                           3 * true_phase.shape[0] // 4, true_phase.shape[0] - 1]))
    figure, axes = plt.subplots(3, len(selected), figsize=(3 * len(selected), 9),
                               squeeze=False, constrained_layout=True)
    for column, index in enumerate(selected):
        target, estimate = true_phase[index], estimated_phase[index]
        bound = max(float(torch.quantile(torch.cat((target.abs().flatten(), estimate.abs().flatten())), 0.99)), 1e-8)
        for row, image in enumerate((target, estimate, (estimate - target).abs())):
            im = axes[row, column].imshow(image, cmap="coolwarm" if row < 2 else "magma",
                                          vmin=-bound if row < 2 else 0,
                                          vmax=bound if row < 2 else max(float(image.max()), 1e-8))
            name = ("True", "Estimated", "Absolute error")[row]
            suffix = " (invisible)" if index == true_phase.shape[0] - 1 else ""
            axes[row, column].set_title(f"{name} phase {index}{suffix}")
            axes[row, column].set_axis_off()
            figure.colorbar(im, ax=axes[row, column], shrink=0.7)
    figure.savefig(output_dir / "phase_screens.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    iterations = [item["iteration"] for item in history]
    axes[0].plot(iterations, [item["data_loss"] for item in history], label="data")
    if any(item["regularization_loss"] > 0 for item in history):
        axes[0].plot(iterations, [item["total_loss"] for item in history], label="total")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot([item["screen"] for item in scores[:-1]],
                 [item["correlation"] for item in scores[:-1]], marker="o")
    axes[1].set_xlabel("Screen index (last excluded)")
    axes[1].set_ylabel("True/estimate correlation")
    axes[1].set_ylim(-1.05, 1.05)
    figure.savefig(output_dir / "optimization.png", dpi=160)
    plt.close(figure)


def invert_one_path(
    model: FNO2d, normalization: Normalization, x: torch.Tensor,
    observed_cpu: torch.Tensor, metadata: dict, chunk_path: Path,
    path_id: int, args: argparse.Namespace, output_dir: Path | None,
) -> dict:
    """Run one inversion; detailed artifacts are optional in test-set mode."""
    device = next(model.parameters()).device
    torch.manual_seed(args.seed + path_id if args.all_test_paths else args.seed)
    if x.shape[-1] != model.input_channels:
        raise ValueError(f"Checkpoint expects {model.input_channels} channels, sample has {x.shape[-1]}")
    rho0 = x[..., 0].to(device)
    observed = observed_cpu.to(device)
    n_intervals = metadata["n_intervals"]
    latent = torch.nn.Parameter(args.init_std * torch.randn((n_intervals, *rho0.shape), device=device))
    optimizer = torch.optim.Adam([latent], lr=args.learning_rate)
    target = normalization.normalize_target(observed)
    history: list[dict] = []
    best_loss = float("inf")
    best_screens = None
    for iteration in range(1, args.max_its + 1):
        optimizer.zero_grad(set_to_none=True)
        screens = latent - latent.mean(dim=(-2, -1), keepdim=True)
        prediction = forward_model(model, normalization, rho0, screens)
        data_loss = (normalization.normalize_target(prediction) - target).square().mean()
        reg_loss = screen_penalty(screens, args.regularization)
        loss = data_loss + args.alpha * reg_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        loss.backward()
        current_loss = float(loss.detach())
        if current_loss < best_loss:
            best_loss = current_loss
            best_screens = screens.detach().clone()
        record = {"iteration": iteration, "data_loss": float(data_loss.detach()),
                  "regularization_loss": float(reg_loss.detach()), "total_loss": current_loss}
        history.append(record)
        optimizer.step()
        if iteration == 1 or iteration % args.print_every == 0 or iteration == args.max_its:
            print(f"FNO iteration {iteration:04d}: data={record['data_loss']:.4e}, total={current_loss:.4e}", flush=True)
    assert best_screens is not None
    best_screens = best_screens.detach().requires_grad_(True)
    best_prediction = forward_model(model, normalization, rho0, best_screens)
    probe = torch.randn_like(best_prediction)
    sensitivity = torch.autograd.grad((best_prediction * probe).mean(), best_screens)[0]
    sensitivity_norms = sensitivity.flatten(1).norm(dim=1).detach().cpu().tolist()
    # Access saved screens only after inversion, for evaluation and plotting.
    true_delta_n = x[..., 1:-1].permute(2, 0, 1).to(device)
    with torch.no_grad():
        true_normalized = true_delta_n / normalization.delta_n_rms
        oracle_prediction = forward_model(model, normalization, rho0, true_normalized)
    quality = image_comparison_metrics(best_prediction, observed)
    oracle_quality = image_comparison_metrics(oracle_prediction, observed)
    dz = metadata["total_distance"] / n_intervals
    k0 = 2 * torch.pi / metadata["wavelength"]
    estimated_delta_n = best_screens.detach() * normalization.delta_n_rms
    true_phase = true_delta_n * k0 * dz
    estimated_phase = estimated_delta_n * k0 * dz
    scores = screen_scores(estimated_phase, true_phase)
    summary = {"method": "fno", "path_id": path_id, "chunk": str(chunk_path),
               "mode_combination_index": metadata.get("mode_combination_index"),
               "checkpoint": str(args.checkpoint), "n_intervals": n_intervals,
               "total_distance_m": metadata["total_distance"], "iterations": args.max_its,
               "best_total_loss": best_loss, "regularization": args.regularization,
               "alpha": args.alpha, "final_image_metrics": quality,
               "negative_prediction_fraction": float((best_prediction < 0).float().mean()),
               "oracle_fno_image_metrics_evaluation_only": oracle_quality,
               "screen_metrics_evaluation_only": scores,
               "probe_sensitivity_gradient_norm_by_screen": sensitivity_norms,
               "last_screen_identifiable_from_rhoZ": False}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_plots(output_dir, rho0, observed, best_prediction, true_phase, estimated_phase, history, scores, quality)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"rho0": rho0.cpu(), "rhoZ_observed": observed.cpu(),
                    "rhoZ_predicted": best_prediction.detach().cpu(),
                    "delta_n_estimated": estimated_delta_n.cpu(),
                    "delta_n_true_evaluation_only": true_delta_n.cpu(),
                    "phase_estimated": estimated_phase.cpu(),
                    "phase_true_evaluation_only": true_phase.cpu()}, output_dir / "fields.pt")
    print(f"FNO path {path_id}: final relative L2={quality['relative_l2']:.4e}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.max_its < 1 or args.learning_rate <= 0 or args.init_std < 0 or args.alpha < 0 or args.print_every < 1:
        raise ValueError("Iterations, learning rate, and print interval must be positive; init std and alpha nonnegative")
    if args.example_plots < 0:
        raise ValueError("--example-plots cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    normalization = Normalization.from_state_dict(checkpoint["normalization"])
    if normalization.delta_n_rms <= 0:
        raise ValueError("Checkpoint delta_n_rms must be positive")
    model = FNO2d(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    manifest = args.checkpoint.parent / "split_manifest.json"
    if not args.all_test_paths:
        if args.resume or args.max_paths is not None:
            raise ValueError("--resume and --max-paths require --all-test-paths")
        path_id = select_path_id(manifest, args.split, args.path_id)
        x, y, metadata, chunk_path = load_final_example(args.data_dir, args.chunk_pattern, path_id)
        output_dir = args.output_dir or args.checkpoint.parent / f"fno_screen_inversion_path_{path_id}"
        invert_one_path(model, normalization, x, y, metadata, chunk_path, path_id, args, output_dir)
        return
    if args.path_id is not None or args.split != "test":
        raise ValueError("--all-test-paths uses the whole test split; omit --path-id and --split")
    selected = test_path_ids(manifest, args.max_paths)
    output_dir = args.output_dir or args.checkpoint.parent / "fno_screen_test_set"
    config = {"method": "fno_screen", "checkpoint": str(args.checkpoint.resolve()),
              "data_dir": str(args.data_dir.resolve()), "chunk_pattern": args.chunk_pattern,
              "max_its": args.max_its, "learning_rate": args.learning_rate,
              "init_std": args.init_std, "regularization": args.regularization,
              "alpha": args.alpha, "seed": args.seed, "device": str(device)}
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
        record = invert_one_path(model, normalization, x, y, metadata, chunk_path,
                                 path_id, args, example_dir)
        append_path_metrics(rows_path, record)
        records.append(record)
        print(f"Completed {len(records)}/{len(selected)} held-out FNO screen inversions", flush=True)
    summary = summarize_screen_records(records, "oracle_fno_image_metrics_evaluation_only")
    summary["method"] = "fno_screen"
    (output_dir / "test_set_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_screen_test_set_plot(output_dir / "test_set_summary.png", records, summary,
                              "FNO screen inversion: held-out paths")
    print(f"Test set complete: {len(records)} paths; summary: {output_dir / 'test_set_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
