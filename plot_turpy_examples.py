import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot representative trajectories from one TurPy chunk."
    )
    parser.add_argument("--chunk", required=True)
    parser.add_argument("--output", default="turpy_examples.png")
    parser.add_argument("--n-examples", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = torch.load(args.chunk, map_location="cpu")

    path_ids = dataset["path_ids"]
    unique_paths = torch.unique(path_ids, sorted=True)
    n_examples = min(args.n_examples, unique_paths.numel())

    selected_positions = torch.linspace(
        0,
        unique_paths.numel() - 1,
        steps=n_examples,
    ).round().long()
    selected_paths = unique_paths[selected_positions]

    figure, axes = plt.subplots(
        n_examples,
        4,
        figsize=(14, 3.5 * n_examples),
        squeeze=False,
    )

    for row, (path_position, path_id) in enumerate(
        zip(selected_positions, selected_paths)
    ):
        sample_indices = torch.nonzero(
            path_ids == path_id,
            as_tuple=True,
        )[0]

        first_index = sample_indices[0]
        middle_index = sample_indices[len(sample_indices) // 2]
        final_index = sample_indices[-1]

        rho0 = dataset["X"][first_index, ..., 0]
        delta_n0 = dataset["X"][first_index, ..., 1]
        rho_middle = dataset["Y"][middle_index, ..., 0]
        rho_final = dataset["Y"][final_index, ..., 0]

        intensity_images = (rho0, rho_middle, rho_final)
        log_intensities = [
            torch.log10(image.clamp_min(1e-12))
            for image in intensity_images
        ]
        intensity_min = min(float(image.min()) for image in log_intensities)
        intensity_max = max(float(image.max()) for image in log_intensities)

        axes[row, 0].imshow(
            log_intensities[0],
            cmap="inferno",
            vmin=intensity_min,
            vmax=intensity_max,
        )

        delta_limit = float(delta_n0.abs().max())
        axes[row, 1].imshow(
            delta_n0,
            cmap="RdBu_r",
            vmin=-delta_limit,
            vmax=delta_limit,
        )

        axes[row, 2].imshow(
            log_intensities[1],
            cmap="inferno",
            vmin=intensity_min,
            vmax=intensity_max,
        )
        axes[row, 3].imshow(
            log_intensities[2],
            cmap="inferno",
            vmin=intensity_min,
            vmax=intensity_max,
        )

        metadata = dataset["path_metadata"][int(path_position)]
        mode_orders = metadata.get(
            "bessel_orders",
            metadata["initial"].get("bessel_orders", ()),
        )
        axes[row, 0].set_ylabel(
            f"path {int(path_id)}\nmodes {tuple(mode_orders)}"
        )

    column_titles = (
        "log10 rho(0)",
        "delta_n(0)",
        "log10 rho(mid)",
        "log10 rho(Z)",
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
