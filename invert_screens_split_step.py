"""Infer phase screens from rho(0) and rho(Z) with differentiable TurPy propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from inversion_evaluation import (
    append_path_metrics, begin_test_set_run, iter_final_test_examples,
    save_screen_test_set_plot, summarize_screen_records, test_path_ids,
)
from turpy import make_turpy_simulator
from utilities import image_comparison_metrics, load_turpy_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("checkpoints/turpy_fno_4km_ic_split/split_manifest.json"),
                        help="Split manifest for selecting the same held-out path as the FNO test.")
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
    parser.add_argument("--init-std", type=float, default=0.05, help="Initial phase-screen standard deviation in radians.")
    parser.add_argument("--regularization", choices=("none", "l2", "smooth"), default="smooth")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--zero-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--padding-factor", type=int, default=2,
                        help="Must match the data generator; old chunks do not record this option.")
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
                    "wavelength": float(chunk["wavelength"]), "n0": float(chunk["n0"]),
                    "dx": float(chunk["dx"]), "initial_phase": chunk.get("initial_phase")}
        if x.shape[-1] != n_intervals + 2:
            raise ValueError("Saved input channels do not match the number of intervals")
        return x, y, metadata, path
    raise ValueError(f"Path ID {path_id} was not found in {data_dir}")


def make_propagator(simulator, height: int, width: int, dz: float,
                    zero_padding: bool, padding_factor: int, device: torch.device):
    if zero_padding and padding_factor > 1:
        padded_height, padded_width = height * padding_factor, width * padding_factor
        top = (padded_height - height) // 2
        left = (padded_width - width) // 2
        right = padded_width - width - left
        bottom = padded_height - height - top
        fx = torch.fft.fftshift(torch.fft.fftfreq(padded_width, d=simulator.dx, device=device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(padded_height, d=simulator.dx, device=device))
        grid_x, grid_y = torch.meshgrid(fx, fy, indexing="xy")
        transfer = torch.exp(-1j * dz * torch.pi * simulator.params["wavelength"]
                             / simulator.params["n"] * (grid_x.square() + grid_y.square()))

        def propagate(field: torch.Tensor) -> torch.Tensor:
            padded = F.pad(field, (left, right, top, bottom))
            return simulator.prop_step(padded, transfer)[top:top + height, left:left + width]

    else:
        transfer = torch.exp(-1j * dz * simulator.sqrt_term)

        def propagate(field: torch.Tensor) -> torch.Tensor:
            return simulator.prop_step(field, transfer)

    return propagate


def final_intensity(rho0: torch.Tensor, phase_screens: torch.Tensor, propagate) -> torch.Tensor:
    # This exactly follows generate_turpy_trajectory: propagate, apply screen,
    # then measure intensity. The final screen cannot alter this measurement.
    field = torch.sqrt(rho0.clamp_min(0)).to(torch.complex64)
    for phase in phase_screens:
        field = propagate(field)
        field = field * torch.exp(1j * phase)
    return field.abs().square()


def screen_penalty(phase_screens: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "l2":
        return phase_screens.square().mean()
    if kind == "smooth":
        return ((phase_screens[:, 1:, :] - phase_screens[:, :-1, :]).square().mean()
                + (phase_screens[:, :, 1:] - phase_screens[:, :, :-1]).square().mean())
    return phase_screens.new_zeros(())


def screen_scores(estimated: torch.Tensor, true: torch.Tensor) -> list[dict]:
    scores = []
    for index, (estimate, reference) in enumerate(zip(estimated, true)):
        estimate = estimate - estimate.mean()
        reference = reference - reference.mean()
        denom = estimate.norm() * reference.norm()
        scores.append({"screen": index, "identifiable_from_rhoZ": index < true.shape[0] - 1,
                       "relative_l2": float((estimate - reference).norm() / reference.norm().clamp_min(1e-12)),
                       "correlation": float((estimate * reference).sum() / denom.clamp_min(1e-12)),
                       "phasor_rmse": float((torch.exp(1j * estimate) - torch.exp(1j * reference)).abs().square().mean().sqrt())})
    return scores


def save_plots(output_dir: Path, rho0: torch.Tensor, observed: torch.Tensor,
               predicted: torch.Tensor, true_phase: torch.Tensor, estimated_phase: torch.Tensor,
               history: list[dict], scores: list[dict], quality: dict) -> None:
    rho0, observed, predicted = [value.detach().cpu() for value in (rho0, observed, predicted)]
    true_phase, estimated_phase = [value.detach().cpu() for value in (true_phase, estimated_phase)]
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    images = [rho0, observed, predicted, (predicted - observed).abs()]
    titles = ["Known rho(0)", "Observed rho(Z)",
              f"Split-step fit\nSSIM={quality['ssim']:.3f}, PSNR={quality['psnr_db']:.1f} dB",
              f"Absolute error\nrel L2={quality['relative_l2']:.3e}"]
    upper = max(float(torch.quantile(torch.cat((observed.flatten(), predicted.flatten())), 0.99)), 1e-12)
    for index, axis in enumerate(axes.flat):
        limit = upper if index < 3 else max(float(torch.quantile(images[index].flatten(), 0.99)), 1e-12)
        im = axis.imshow(images[index], cmap="inferno", vmin=0, vmax=limit)
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
            im = axes[row, column].imshow(image, cmap="RdBu_r" if row < 2 else "magma",
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
    axes[0].set_ylabel("Relative squared-image loss")
    axes[0].legend()
    axes[1].plot([item["screen"] for item in scores[:-1]],
                 [item["correlation"] for item in scores[:-1]], marker="o")
    axes[1].set_xlabel("Screen index (last excluded)")
    axes[1].set_ylabel("True/estimate correlation")
    axes[1].set_ylim(-1.05, 1.05)
    figure.savefig(output_dir / "optimization.png", dpi=160)
    plt.close(figure)


def configured_propagator(metadata: dict, shape: tuple[int, int],
                          args: argparse.Namespace, device: torch.device):
    height, width = shape
    if height != width:
        raise ValueError("The current TurPy simulator adapter requires a square grid")
    dz = metadata["total_distance"] / metadata["n_intervals"]
    _, simulator = make_turpy_simulator(grid_size=height, dx=metadata["dx"],
                                        wavelength=metadata["wavelength"], n0=metadata["n0"],
                                        device=str(device))
    return make_propagator(simulator, height, width, dz, args.zero_padding,
                           args.padding_factor, device)


def invert_one_path(
    x: torch.Tensor, observed_cpu: torch.Tensor, metadata: dict,
    chunk_path: Path, path_id: int, args: argparse.Namespace, device: torch.device,
    propagate, output_dir: Path | None,
) -> dict:
    """Run one physical inversion, with optional detailed example artifacts."""
    if metadata["initial_phase"] != "flat":
        raise ValueError("Physical inversion requires a flat initial phase; rho0 alone cannot reconstruct a vortex initial field")
    height, width = observed_cpu.shape
    if height != width:
        raise ValueError("The current TurPy simulator adapter requires a square grid")
    torch.manual_seed(args.seed + path_id if args.all_test_paths else args.seed)
    rho0 = x[..., 0].to(device)
    observed = observed_cpu.to(device)
    n_intervals = metadata["n_intervals"]
    dz = metadata["total_distance"] / n_intervals
    k0 = 2 * torch.pi / metadata["wavelength"]
    latent = torch.nn.Parameter(args.init_std * torch.randn((n_intervals, height, width), device=device))
    optimizer = torch.optim.Adam([latent], lr=args.learning_rate)
    intensity_scale = observed.square().mean().clamp_min(1e-20)
    history: list[dict] = []
    best_loss = float("inf")
    best_screens = None
    for iteration in range(1, args.max_its + 1):
        optimizer.zero_grad(set_to_none=True)
        phase = latent - latent.mean(dim=(-2, -1), keepdim=True)
        prediction = final_intensity(rho0, phase, propagate)
        data_loss = (prediction - observed).square().mean() / intensity_scale
        reg_loss = screen_penalty(phase, args.regularization)
        loss = data_loss + args.alpha * reg_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        loss.backward()
        current_loss = float(loss.detach())
        if current_loss < best_loss:
            best_loss = current_loss
            best_screens = phase.detach().clone()
        record = {"iteration": iteration, "data_loss": float(data_loss.detach()),
                  "regularization_loss": float(reg_loss.detach()), "total_loss": current_loss}
        history.append(record)
        optimizer.step()
        if iteration == 1 or iteration % args.print_every == 0 or iteration == args.max_its:
            print(f"Split-step iteration {iteration:04d}: data={record['data_loss']:.4e}, total={current_loss:.4e}", flush=True)
    assert best_screens is not None
    best_screens = best_screens.detach().requires_grad_(True)
    best_prediction = final_intensity(rho0, best_screens, propagate)
    probe = torch.randn_like(best_prediction)
    sensitivity = torch.autograd.grad((best_prediction * probe).mean(), best_screens)[0]
    sensitivity_norms = sensitivity.flatten(1).norm(dim=1).detach().cpu().tolist()
    # Access saved screens only after inversion, for evaluation and plotting.
    true_delta_n = x[..., 1:-1].permute(2, 0, 1).to(device)
    true_phase = true_delta_n * k0 * dz
    with torch.no_grad():
        oracle_prediction = final_intensity(rho0, true_phase, propagate)
    quality = image_comparison_metrics(best_prediction, observed)
    oracle_quality = image_comparison_metrics(oracle_prediction, observed)
    scores = screen_scores(best_screens.detach(), true_phase)
    summary = {"method": "split_step", "path_id": path_id, "chunk": str(chunk_path),
               "mode_combination_index": metadata.get("mode_combination_index"),
               "n_intervals": n_intervals, "total_distance_m": metadata["total_distance"],
               "zero_padding": args.zero_padding, "padding_factor": args.padding_factor,
               "iterations": args.max_its, "best_total_loss": best_loss,
               "regularization": args.regularization, "alpha": args.alpha,
               "final_image_metrics": quality,
               "oracle_split_step_image_metrics_evaluation_only": oracle_quality,
               "screen_metrics_evaluation_only": scores,
               "probe_sensitivity_gradient_norm_by_screen": sensitivity_norms,
               "last_screen_identifiable_from_rhoZ": False}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_plots(output_dir, rho0, observed, best_prediction, true_phase,
                   best_screens, history, scores, quality)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"rho0": rho0.cpu(), "rhoZ_observed": observed.cpu(),
                    "rhoZ_predicted": best_prediction.detach().cpu(),
                    "delta_n_estimated": (best_screens.detach() / (k0 * dz)).cpu(),
                    "delta_n_true_evaluation_only": true_delta_n.cpu(),
                    "phase_estimated": best_screens.detach().cpu(),
                    "phase_true_evaluation_only": true_phase.cpu()}, output_dir / "fields.pt")
    if oracle_quality["relative_l2"] > 1e-3:
        print(f"WARNING path {path_id}: true-screen forward mismatch is {oracle_quality['relative_l2']:.3e}; check padding settings and data provenance", flush=True)
    print(f"Split-step path {path_id}: final relative L2={quality['relative_l2']:.4e}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.max_its < 1 or args.learning_rate <= 0 or args.init_std < 0 or args.alpha < 0 or args.print_every < 1:
        raise ValueError("Iterations, learning rate, and print interval must be positive; init std and alpha nonnegative")
    if args.padding_factor < 1 or args.example_plots < 0:
        raise ValueError("Padding factor must be positive and example-plots nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if not args.all_test_paths:
        if args.resume or args.max_paths is not None:
            raise ValueError("--resume and --max-paths require --all-test-paths")
        path_id = select_path_id(args.manifest, args.split, args.path_id)
        x, y, metadata, chunk_path = load_final_example(args.data_dir, args.chunk_pattern, path_id)
        propagate = configured_propagator(metadata, y.shape, args, device)
        output_dir = args.output_dir or args.manifest.parent / f"split_step_screen_inversion_path_{path_id}"
        invert_one_path(x, y, metadata, chunk_path, path_id, args, device, propagate, output_dir)
        return
    if args.path_id is not None or args.split != "test":
        raise ValueError("--all-test-paths uses the whole test split; omit --path-id and --split")
    selected = test_path_ids(args.manifest, args.max_paths)
    output_dir = args.output_dir or args.manifest.parent / "split_step_screen_test_set"
    config = {"method": "split_step_screen", "manifest": str(args.manifest.resolve()),
              "data_dir": str(args.data_dir.resolve()), "chunk_pattern": args.chunk_pattern,
              "max_its": args.max_its, "learning_rate": args.learning_rate,
              "init_std": args.init_std, "regularization": args.regularization,
              "alpha": args.alpha, "seed": args.seed, "device": str(device),
              "zero_padding": args.zero_padding, "padding_factor": args.padding_factor}
    records, rows_path = begin_test_set_run(output_dir, config, args.resume)
    completed = {row["path_id"] for row in records}
    if completed - set(selected):
        raise ValueError("Existing metrics include paths outside the selected test set")
    example_ids = set(selected[:args.example_plots])
    propagators = {}
    for path_id, x, y, metadata, chunk_path in iter_final_test_examples(
        args.data_dir, args.chunk_pattern, selected
    ):
        if path_id in completed:
            continue
        key = (tuple(y.shape), metadata["n_intervals"], metadata["total_distance"],
               metadata["dx"], metadata["wavelength"], metadata["n0"])
        if key not in propagators:
            propagators[key] = configured_propagator(metadata, y.shape, args, device)
        example_dir = output_dir / "examples" / f"path_{path_id}" if path_id in example_ids else None
        record = invert_one_path(x, y, metadata, chunk_path, path_id, args,
                                 device, propagators[key], example_dir)
        append_path_metrics(rows_path, record)
        records.append(record)
        print(f"Completed {len(records)}/{len(selected)} held-out split-step screen inversions", flush=True)
    summary = summarize_screen_records(records, "oracle_split_step_image_metrics_evaluation_only")
    summary["method"] = "split_step_screen"
    (output_dir / "test_set_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_screen_test_set_plot(output_dir / "test_set_summary.png", records, summary,
                              "Split-step screen inversion: held-out paths")
    print(f"Test set complete: {len(records)} paths; summary: {output_dir / 'test_set_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
