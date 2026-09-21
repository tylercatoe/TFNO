import torch
import numpy as np
import argparse
from pathlib import Path


def make_random_initial_field(
    xx,
    yy,
    k,
    waist_range=(5e-4, 1.2e-3),
    tilt_range=(-0.005, 0.005),
    curvature_range=(100.0, 1000.0),
    center_fraction=0.2,
):
    """
    Create a randomized initial complex optical field.

    Randomizes:
        - Beam waist
        - Beam center
        - Amplitude
        - Wavefront tilt
        - Wavefront curvature
    """

    device = xx.device

    x_extent = xx.abs().max()
    y_extent = yy.abs().max()

    beam_waist = torch.empty(
        1,
        device=device,
    ).uniform_(
        waist_range[0],
        waist_range[1],
    ).item()

    x_center = torch.empty(
        1,
        device=device,
    ).uniform_(
        -center_fraction * x_extent,
        center_fraction * x_extent,
    ).item()

    y_center = torch.empty(
        1,
        device=device,
    ).uniform_(
        -center_fraction * y_extent,
        center_fraction * y_extent,
    ).item()

    amplitude_scale = torch.empty(
        1,
        device=device,
    ).uniform_(0.8, 1.2).item()

    theta_x = torch.empty(
        1,
        device=device,
    ).uniform_(
        tilt_range[0],
        tilt_range[1],
    ).item()

    theta_y = torch.empty(
        1,
        device=device,
    ).uniform_(
        tilt_range[0],
        tilt_range[1],
    ).item()

    radius = torch.empty(
        1,
        device=device,
    ).uniform_(
        curvature_range[0],
        curvature_range[1],
    ).item()

    x_shifted = xx - x_center
    y_shifted = yy - y_center

    amplitude = amplitude_scale * torch.exp(
        -(
            x_shifted**2
            + y_shifted**2
        ) / beam_waist**2
    )

    initial_phase = (
        k * theta_x * x_shifted
        + k * theta_y * y_shifted
        + k * (
            x_shifted**2
            + y_shifted**2
        ) / (2.0 * radius)
    )

    initial_field = amplitude * torch.exp(
        1j * initial_phase
    )

    metadata = {
        "beam_waist": beam_waist,
        "x_center": x_center,
        "y_center": y_center,
        "amplitude_scale": amplitude_scale,
        "theta_x": theta_x,
        "theta_y": theta_y,
        "radius": radius,
    }

    return initial_field, metadata


@torch.no_grad()
def generate_turpy_trajectory(
    simulator,
    params,
    n_z=21,
    total_distance=1000.0,
    r0_min=0.03,
    r0_max=0.15,
    seed=None,
):
    """
    Generate one complete TurPy propagation trajectory.

    Each delta-n screen represents one interval between adjacent
    z-grid points.

    Returns:
        initial_field:
            Complex tensor with shape [H, W]

        delta_n:
            Tensor with shape [n_z - 1, H, W]

        intensities:
            Tensor with shape [n_z, H, W]

        r0:
            Fried parameters for each interval

        dz:
            Uniform propagation spacing
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = params["device"]

    xx = simulator.xx
    yy = simulator.yy

    n_intervals = n_z - 1
    dz = total_distance / n_intervals

    # TurPy uses k = 2*pi*n/lambda
    k = float(params["k"])

    # Randomized initial complex field
    initial_field, initial_metadata = make_random_initial_field(
        xx=xx,
        yy=yy,
        k=k,
    )

    field = initial_field.clone()

    # Store intensity at z = 0
    intensities = [
        torch.abs(field) ** 2
    ]

    delta_n_screens = []

    # Turbulence strength for each interval
    r0_values = torch.empty(
        n_intervals,
        device=device,
        dtype=torch.float32,
    ).uniform_(
        r0_min,
        r0_max,
    )

    # Uniform propagation transfer function
    transfer_function = torch.exp(
        1j * dz * simulator.sqrt_term
    )

    for j in range(n_intervals):

        # Propagate from z[j] to z[j+1]
        field = simulator.prop_step(
            field,
            transfer_function,
        )

        # TurPy samples a phase screen phi
        phase_screen = simulator.phase_screen.sample(
            r0_values[j],
            seed=None if seed is None else seed + j + 1,
        )

        # Thin-slab relation:
        #
        # phi = k * delta_n * dz
        delta_n = phase_screen / (k * dz)

        # Apply the refractive-index perturbation
        field = field * torch.exp(
            1j * k * delta_n * dz
        )

        delta_n_screens.append(delta_n)

        # Store intensity at the new z-grid point
        intensities.append(
            torch.abs(field) ** 2
        )

    return {
        "initial_field": initial_field.cpu(),
        "delta_n": torch.stack(
            delta_n_screens
        ).cpu(),
        "intensities": torch.stack(
            intensities
        ).cpu(),
        "r0": r0_values.cpu(),
        "dz": dz,
        "initial_metadata": initial_metadata,
    }


def trajectory_to_one_step_examples(
    trajectory,
):
    """
    Convert one propagation trajectory into one-step examples.

    Example j predicts intensity at z[j+1].

    Inputs contain:
        - Initial intensity
        - Initial phase encoded as cosine and sine
        - True delta-n history through interval j
        - History mask
        - Target z position

    Returns:
        X:
            [n_intervals, H, W, n_channels]

        Y:
            [n_intervals, H, W, 1]
    """

    initial_field = trajectory["initial_field"]
    delta_n = trajectory["delta_n"]
    intensities = trajectory["intensities"]

    n_intervals, H, W = delta_n.shape

    initial_intensity = torch.abs(
        initial_field
    ) ** 2

    initial_phase = torch.angle(
        initial_field
    )

    # Channels:
    #
    # 0: initial intensity
    # 1: cos(initial phase)
    # 2: sin(initial phase)
    # 3 ... 3+n_intervals-1:
    #     padded delta-n screens
    #
    # next n_intervals:
    #     history mask
    #
    # final channel:
    #     normalized target z
    n_channels = 3 + 2 * n_intervals + 1

    X = torch.zeros(
        n_intervals,
        H,
        W,
        n_channels,
        dtype=torch.float32,
    )

    Y = torch.zeros(
        n_intervals,
        H,
        W,
        1,
        dtype=torch.float32,
    )

    mask_start = 3 + n_intervals
    z_channel = n_channels - 1

    for j in range(n_intervals):

        # Initial condition channels
        X[j, ..., 0] = initial_intensity
        X[j, ..., 1] = torch.cos(
            initial_phase
        )
        X[j, ..., 2] = torch.sin(
            initial_phase
        )

        # True delta-n history through interval j
        X[
            j,
            ...,
            3:3 + j + 1
        ] = delta_n[
            :j + 1
        ].permute(1, 2, 0)

        # History mask
        X[
            j,
            ...,
            mask_start:mask_start + j + 1
        ] = 1.0

        # Target z position normalized to [0, 1]
        X[
            j,
            ...,
            z_channel
        ] = (j + 1) / n_intervals

        # Target intensity at z[j+1]
        Y[
            j,
            ...,
            0
        ] = intensities[j + 1]

    return X, Y


def make_one_step_dataset(
    simulator,
    params,
    n_paths=100,
    n_z=21,
    total_distance=5000.0,
    r0_min=0.03,
    r0_max=0.15,
    seed=123,
    path_start=0,
):
    """
    Generate a complete in-memory dataset efficiently.

    Each propagation path creates n_z - 1 examples.

    Returns one dictionary containing:
        X
        Y
        path_ids
        path_metadata
        n_z
        total_distance
    """

    n_steps = n_z - 1
    n_samples = n_paths * n_steps

    H, W = params["field_size"]

    # 3 initial-condition channels
    # n_steps delta-n channels
    # n_steps history-mask channels
    # 1 target-z channel
    n_channels = 3 + 2 * n_steps + 1

    # Preallocate tensors to avoid list concatenation
    X = torch.empty(
        n_samples,
        H,
        W,
        n_channels,
        dtype=torch.float32,
    )

    Y = torch.empty(
        n_samples,
        H,
        W,
        1,
        dtype=torch.float32,
    )

    path_ids = torch.empty(
        n_samples,
        dtype=torch.long,
    )

    path_metadata = []

    for local_path_id in range(n_paths):

        path_id = path_start + local_path_id

        trajectory = generate_turpy_trajectory(
            simulator=simulator,
            params=params,
            n_z=n_z,
            total_distance=total_distance,
            r0_min=r0_min,
            r0_max=r0_max,
            seed=seed + path_id,
        )

        X_path, Y_path = (
            trajectory_to_one_step_examples(
                trajectory
            )
        )

        start = local_path_id * n_steps
        end = start + n_steps

        X[start:end] = X_path
        Y[start:end] = Y_path
        path_ids[start:end] = path_id

        path_metadata.append(
            {
                "r0": trajectory["r0"],
                "dz": trajectory["dz"],
                "initial": trajectory[
                    "initial_metadata"
                ],
            }
        )

        if (local_path_id + 1) % 10 == 0:
            print(
                f"Generated "
                f"{local_path_id + 1}/{n_paths} paths"
            )

    return {
        "X": X,
        "Y": Y,
        "path_ids": path_ids,
        "path_metadata": path_metadata,
        "n_z": n_z,
        "total_distance": total_distance,
        "path_start": path_start,
    }


def make_path_splits(
    dataset,
    train_fraction=0.8,
    val_fraction=0.1,
    seed=123,
):
    """
    Split by complete propagation path.

    This prevents windows from the same physical path
    appearing in both training and validation/test sets.
    """

    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be less than 1.0"
        )

    path_ids = dataset["path_ids"]
    unique_paths = torch.unique(path_ids)

    generator = torch.Generator().manual_seed(
        seed
    )

    permutation = torch.randperm(
        unique_paths.numel(),
        generator=generator,
    )

    shuffled_paths = unique_paths[
        permutation
    ]

    n_paths = shuffled_paths.numel()
    n_train = int(train_fraction * n_paths)
    n_val = int(val_fraction * n_paths)

    train_paths = shuffled_paths[
        :n_train
    ]

    val_paths = shuffled_paths[
        n_train:n_train + n_val
    ]

    test_paths = shuffled_paths[
        n_train + n_val:
    ]

    train_idx = torch.isin(
        path_ids,
        train_paths,
    ).nonzero(
        as_tuple=True
    )[0]

    val_idx = torch.isin(
        path_ids,
        val_paths,
    ).nonzero(
        as_tuple=True
    )[0]

    test_idx = torch.isin(
        path_ids,
        test_paths,
    ).nonzero(
        as_tuple=True
    )[0]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def save_dataset_chunk(
    output_path,
    *,
    grid_size=64,
    dx=20e-6,
    subharmonics=True,
    n_paths=100,
    n_z=21,
    total_distance=5000.0,
    r0_min=0.03,
    r0_max=0.15,
    seed=123,
    path_start=0,
):
    """Generate and save one independent HPC chunk."""

    from turpy import make_turpy_simulator

    params, simulator = make_turpy_simulator(
        grid_size=grid_size,
        dx=dx,
        subharmonics=subharmonics,
    )

    dataset = make_one_step_dataset(
        simulator=simulator,
        params=params,
        n_paths=n_paths,
        n_z=n_z,
        total_distance=total_distance,
        r0_min=r0_min,
        r0_max=r0_max,
        seed=seed,
        path_start=path_start,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\nSaved chunk: {output_path}")
    print(f"X: {dataset['X'].shape}")
    print(f"Y: {dataset['Y'].shape}")
    print(
        f"Paths: {path_start}--"
        f"{path_start + n_paths - 1}"
    )


def merge_dataset_chunks(
    chunk_dir,
    output_path,
    *,
    train_fraction=0.8,
    val_fraction=0.1,
    split_seed=123,
):
    """Merge chunk files and create one global path-based split."""

    chunk_dir = Path(chunk_dir)
    output_path = Path(output_path)
    chunk_paths = sorted(
        path for path in chunk_dir.glob("*.pt")
        if path.resolve() != output_path.resolve()
    )

    if not chunk_paths:
        raise FileNotFoundError(
            f"No .pt chunk files found in {chunk_dir}"
        )

    chunks = [
        torch.load(path, map_location="cpu")
        for path in chunk_paths
    ]

    reference = chunks[0]
    for path, chunk in zip(chunk_paths[1:], chunks[1:]):
        if chunk["n_z"] != reference["n_z"]:
            raise ValueError(
                f"n_z mismatch in {path}: "
                f"{chunk['n_z']} != {reference['n_z']}"
            )
        if chunk["X"].shape[1:] != reference["X"].shape[1:]:
            raise ValueError(
                f"Tensor shape mismatch in {path}: "
                f"{chunk['X'].shape[1:]} != "
                f"{reference['X'].shape[1:]}"
            )
        if chunk["total_distance"] != reference["total_distance"]:
            raise ValueError(
                f"total_distance mismatch in {path}: "
                f"{chunk['total_distance']} != "
                f"{reference['total_distance']}"
            )

    all_path_ids = torch.cat(
        [chunk["path_ids"] for chunk in chunks]
    )
    expected_unique_paths = sum(
        torch.unique(chunk["path_ids"]).numel()
        for chunk in chunks
    )
    if torch.unique(all_path_ids).numel() != expected_unique_paths:
        raise ValueError(
            "Overlapping path IDs detected across chunks. "
            "Use a unique --path-start for every HPC job."
        )

    dataset = {
        "X": torch.cat([chunk["X"] for chunk in chunks]),
        "Y": torch.cat([chunk["Y"] for chunk in chunks]),
        "path_ids": torch.cat(
            [chunk["path_ids"] for chunk in chunks]
        ),
        "path_metadata": [
            metadata
            for chunk in chunks
            for metadata in chunk["path_metadata"]
        ],
        "n_z": reference["n_z"],
        "total_distance": reference["total_distance"],
    }

    dataset["splits"] = make_path_splits(
        dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=split_seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    print(f"\nMerged {len(chunk_paths)} chunks into: {output_path}")
    print(f"X: {dataset['X'].shape}")
    print(f"Y: {dataset['Y'].shape}")
    for split_name in ("train", "val", "test"):
        split_indices = dataset["splits"][f"{split_name}_idx"]
        print(f"{split_name.capitalize()}: {split_indices.numel()} samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate or merge TurPy training-data chunks."
    )
    parser.add_argument(
        "--mode",
        choices=("generate", "merge"),
        default="generate",
    )
    parser.add_argument("--output", default="turpy_step_dataset.pt")
    parser.add_argument("--chunk-dir", default="turpy_chunks")
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--path-start", type=int, default=0)
    parser.add_argument("--n-z", type=int, default=21)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--dx", type=float, default=20e-6)
    parser.add_argument("--total-distance", type=float, default=5000.0)
    parser.add_argument("--r0-min", type=float, default=0.03)
    parser.add_argument("--r0-max", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument(
        "--no-subharmonics",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "generate":
        save_dataset_chunk(
            args.output,
            grid_size=args.grid_size,
            dx=args.dx,
            subharmonics=not args.no_subharmonics,
            n_paths=args.n_paths,
            n_z=args.n_z,
            total_distance=args.total_distance,
            r0_min=args.r0_min,
            r0_max=args.r0_max,
            seed=args.seed,
            path_start=args.path_start,
        )
    else:
        merge_dataset_chunks(
            args.chunk_dir,
            args.output,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
        )
