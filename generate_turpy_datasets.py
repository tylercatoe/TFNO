import torch
import numpy as np
import argparse
from itertools import combinations
from pathlib import Path


def make_random_initial_field(
    xx,
    yy,
    k,
    dx,
    waist_range=(0.5, 0.5),
    tilt_range=(-0.005, 0.005),
    curvature_range=(100.0, 1000.0),
    center_fraction=0.2,
    centered=True,
    beam_type="gaussian_bessel",
    bessel_orders=(0,),
    bessel_kr=20.0,
    initial_phase_mode="flat",
    normalize_power=True,
    fixed_mode_coefficients=True,
):
    """
    Create an initial complex optical field.

    Can randomize:
        - Beam waist
        - Beam center
        - Amplitude
        - Wavefront tilt
        - Wavefront curvature
    """

    if waist_range[0] <= 0 or waist_range[1] < waist_range[0]:
        raise ValueError("waist_range must contain positive ascending values")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if initial_phase_mode not in ("flat", "vortex"):
        raise ValueError("initial_phase_mode must be 'flat' or 'vortex'")

    device = xx.device

    requested_beam_type = beam_type
    if beam_type == "mixed":
        beam_type = (
            "gaussian_bessel"
            if int(torch.randint(0, 2, (1,), device=device))
            else "gaussian"
        )

    x_extent = xx.abs().max()
    y_extent = yy.abs().max()

    beam_waist = torch.empty(
        1,
        device=device,
    ).uniform_(
        waist_range[0],
        waist_range[1],
    ).item()

    if centered:
        x_center = 0.0
        y_center = 0.0
    else:
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

    amplitude_scale = 1.0 if normalize_power else torch.empty(
        1, device=device
    ).uniform_(0.8, 1.2).item()

    if centered:
        theta_x = 0.0
        theta_y = 0.0
    else:
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

    radius = float("inf") if initial_phase_mode == "flat" else torch.empty(
        1, device=device
    ).uniform_(curvature_range[0], curvature_range[1]).item()

    x_shifted = xx - x_center
    y_shifted = yy - y_center

    radius_grid = torch.sqrt(
        x_shifted**2 + y_shifted**2
    )

    gaussian_envelope = torch.exp(
        -radius_grid**2 / beam_waist**2
    )

    bessel_coefficients = None
    if beam_type == "gaussian":
        amplitude = amplitude_scale * gaussian_envelope
    elif beam_type == "gaussian_bessel":
        if not bessel_orders:
            raise ValueError(
                "bessel_orders must contain at least one mode"
            )

        def bessel_integer(order, argument):
            sign = -1.0 if order < 0 and abs(order) % 2 else 1.0
            order = abs(int(order))

            if order == 0:
                values = torch.special.bessel_j0(argument)
            elif order == 1:
                values = torch.special.bessel_j1(argument)
            else:
                j_previous = torch.special.bessel_j0(argument)
                j_current = torch.special.bessel_j1(argument)
                argument_safe = torch.where(
                    argument.abs() < 1e-12,
                    torch.ones_like(argument),
                    argument,
                )

                for recurrence_order in range(1, order):
                    j_next = (
                        2.0
                        * recurrence_order
                        / argument_safe
                        * j_current
                        - j_previous
                    )
                    j_next = torch.where(
                        argument.abs() < 1e-12,
                        torch.zeros_like(j_next),
                        j_next,
                    )
                    j_previous, j_current = j_current, j_next

                values = j_current

            return sign * values

        if fixed_mode_coefficients:
            bessel_coefficients = torch.ones(
                len(bessel_orders), dtype=torch.cfloat, device=device
            )
        else:
            bessel_coefficients = torch.randn(
                len(bessel_orders), device=device
            ) + 1j * torch.randn(len(bessel_orders), device=device)
        bessel_coefficients = bessel_coefficients / torch.linalg.vector_norm(
            bessel_coefficients
        )

        azimuth = torch.atan2(
            y_shifted,
            x_shifted,
        )
        bessel_superposition = torch.zeros(
            xx.shape,
            dtype=torch.cfloat,
            device=device,
        )

        for coefficient, order in zip(
            bessel_coefficients,
            bessel_orders,
        ):
            radial_mode = bessel_integer(
                order,
                bessel_kr * radius_grid,
            )
            angular_mode = torch.exp(
                1j * float(order) * azimuth
            )
            bessel_superposition += (
                coefficient
                * radial_mode
                * angular_mode
            )

        amplitude = (
            amplitude_scale
            * gaussian_envelope
            * bessel_superposition
        )
    else:
        raise ValueError(
            f"Unknown beam_type: {beam_type}"
        )

    if initial_phase_mode == "flat":
        # The learning input contains rho0 but not the complex phase.  Using
        # sqrt(rho0) makes propagation identifiable from the saved input.
        initial_field = torch.abs(amplitude).to(torch.cfloat)
        theta_x = 0.0
        theta_y = 0.0
    else:
        geometric_phase = (
            k * theta_x * x_shifted
            + k * theta_y * y_shifted
            + k * (x_shifted**2 + y_shifted**2) / (2.0 * radius)
        )
        initial_field = amplitude * torch.exp(1j * geometric_phase)

    initial_power = torch.sum(torch.abs(initial_field) ** 2) * dx**2
    if initial_power <= 0:
        raise RuntimeError("Initial field has zero integrated power")
    if normalize_power:
        initial_field = initial_field / torch.sqrt(initial_power)

    metadata = {
        "beam_type": beam_type,
        "beam_type_requested": requested_beam_type,
        "beam_waist": beam_waist,
        "x_center": x_center,
        "y_center": y_center,
        "amplitude_scale": amplitude_scale,
        "theta_x": theta_x,
        "theta_y": theta_y,
        "radius": radius,
        "initial_phase": initial_phase_mode,
        "normalize_power": normalize_power,
        "fixed_mode_coefficients": fixed_mode_coefficients,
        "integrated_power": float(
            (torch.sum(torch.abs(initial_field) ** 2) * dx**2).cpu()
        ),
    }

    if bessel_coefficients is not None:
        metadata["bessel_orders"] = [
            int(order) for order in bessel_orders
        ]
        metadata["bessel_kr"] = bessel_kr
        metadata["bessel_coefficients"] = [
            [
                float(coefficient.real),
                float(coefficient.imag),
            ]
            for coefficient in bessel_coefficients.cpu()
        ]

    return initial_field, metadata


def fried_parameter_for_segment(cn2, wavelength, dz):
    """Return the Fried parameter for a constant-Cn2 propagation slab."""
    if cn2 <= 0 or wavelength <= 0 or dz <= 0:
        raise ValueError("cn2, wavelength, and dz must be positive")
    k0 = 2.0 * np.pi / wavelength
    return (0.423 * k0**2 * cn2 * dz) ** (-3.0 / 5.0)


def propagate_zero_padded(
    field,
    simulator,
    dz,
    padding_factor=2,
):
    """Propagate on a larger zero-padded grid and crop back to the field size."""

    if padding_factor < 1:
        raise ValueError("padding_factor must be at least 1")

    if padding_factor == 1:
        transfer_function = torch.exp(
            1j * dz * simulator.sqrt_term
        )
        return simulator.prop_step(field, transfer_function)

    height, width = field.shape[-2:]
    padded_height = int(height * padding_factor)
    padded_width = int(width * padding_factor)

    pad_top = (padded_height - height) // 2
    pad_left = (padded_width - width) // 2

    padded_field = torch.zeros(
        padded_height,
        padded_width,
        dtype=field.dtype,
        device=field.device,
    )
    padded_field[
        pad_top:pad_top + height,
        pad_left:pad_left + width,
    ] = field

    fx = torch.fft.fftshift(
        torch.fft.fftfreq(
            padded_width,
            d=simulator.dx,
            device=field.device,
        )
    )
    fy = torch.fft.fftshift(
        torch.fft.fftfreq(
            padded_height,
            d=simulator.dx,
            device=field.device,
        )
    )
    f_x, f_y = torch.meshgrid(fx, fy, indexing="xy")
    f_r_squared = f_x**2 + f_y**2

    transfer_function = torch.exp(
        1j
        * dz
        * (
            torch.pi
            * simulator.params["wavelength"]
            / simulator.params["n"]
        )
        * f_r_squared
    )

    propagated = simulator.prop_step(
        padded_field,
        transfer_function,
    )

    return propagated[
        pad_top:pad_top + height,
        pad_left:pad_left + width,
    ]


@torch.no_grad()
def generate_turpy_trajectory(
    simulator,
    params,
    n_z=21,
    total_distance=2000.0,
    r0_min=0.03,
    r0_max=0.15,
    cn2=1e-15,
    seed=None,
    centered=True,
    zero_padding=True,
    padding_factor=2,
    beam_type="gaussian_bessel",
    bessel_orders=(0,),
    bessel_kr=20.0,
    beam_waist_range=(0.5, 0.5),
    initial_phase="flat",
    normalize_power=True,
    fixed_mode_coefficients=True,
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

    if n_z < 2:
        raise ValueError("n_z must be at least 2")
    if total_distance <= 0:
        raise ValueError("total_distance must be positive")

    n_intervals = n_z - 1
    dz = total_distance / n_intervals

    # Use the vacuum wavenumber for phi = k0 * delta_n * dz, matching
    # the reference split-step generator. TurPy uses wavelength/n0 for
    # its free-space transfer function.
    k0 = 2.0 * np.pi / float(params["wavelength"])

    # Randomized initial complex field
    initial_field, initial_metadata = make_random_initial_field(
        xx=xx,
        yy=yy,
        k=float(params["k"]),
        dx=float(params["dx"]),
        waist_range=beam_waist_range,
        centered=centered,
        beam_type=beam_type,
        bessel_orders=bessel_orders,
        bessel_kr=bessel_kr,
        initial_phase_mode=initial_phase,
        normalize_power=normalize_power,
        fixed_mode_coefficients=fixed_mode_coefficients,
    )

    field = initial_field.clone()

    # Store intensity at z = 0
    intensities = [
        torch.abs(field) ** 2
    ]

    delta_n_screens = []

    # Cn2 defines a fixed Fried parameter for each equal-length slab. Keep
    # the old random r0 range available when cn2=None.
    if cn2 is None:
        r0_values = torch.empty(
            n_intervals, device=device, dtype=torch.float32
        ).uniform_(r0_min, r0_max)
    else:
        r0_segment = fried_parameter_for_segment(
            cn2=cn2,
            wavelength=float(params["wavelength"]),
            dz=dz,
        )
        r0_values = torch.full(
            (n_intervals,), r0_segment, device=device, dtype=torch.float32
        )

    for j in range(n_intervals):

        # Propagate from z[j] to z[j+1]
        if zero_padding:
            field = propagate_zero_padded(
                field,
                simulator,
                dz,
                padding_factor=padding_factor,
            )
        else:
            transfer_function = torch.exp(
                1j * dz * simulator.sqrt_term
            )
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
        # phi = k0 * delta_n * dz
        delta_n = phase_screen / (k0 * dz)

        # Apply the refractive-index perturbation
        field = field * torch.exp(
            1j * k0 * delta_n * dz
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
        "cn2": cn2,
        "dz": dz,
        "initial_metadata": initial_metadata,
    }


def trajectory_to_one_step_examples(
    trajectory,
):
    """
    Convert one propagation trajectory into one-step examples.

    Example j predicts intensity at z[j+1].

    Inputs contain, in channel order:
        - rho0: initial intensity
        - one slot per interval, containing either delta-n[k] or
          a mask value for an unavailable future screen
        - one history-fraction channel

    For sample j, slots 0 through j contain delta-n screens and
    slots j+1 through n_intervals-1 are zero-filled masks.

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
    # 0: rho0
    # 1 ... n_intervals: delta-n-or-mask slots
    # final channel: fraction of the delta-n history available
    n_channels = 2 + n_intervals

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

    history_fraction_channel = 1 + n_intervals

    for j in range(n_intervals):

        # Initial intensity channel rho0
        X[j, ..., 0] = initial_intensity

        # True delta-n history through interval j
        X[
            j,
            ...,
            1:1 + j + 1
        ] = delta_n[
            :j + 1
        ].permute(1, 2, 0)

        # History fraction: (j + 1) / n_intervals
        X[
            j,
            ...,
            history_fraction_channel
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
    total_distance=2000.0,
    r0_min=0.03,
    r0_max=0.15,
    cn2=1e-15,
    seed=123,
    path_start=0,
    centered=True,
    zero_padding=True,
    padding_factor=2,
    beam_type="gaussian_bessel",
    bessel_orders=(0,),
    bessel_kr=20.0,
    bessel_order_pool=None,
    beam_waist_range=(0.5, 0.5),
    initial_phase="flat",
    normalize_power=True,
    fixed_mode_coefficients=True,
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

    # 1 rho0 channel
    # n_steps delta-n channels
    # 1 history-fraction channel
    n_channels = 2 + n_steps

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

    if bessel_order_pool is not None:
        if beam_type == "gaussian":
            raise ValueError(
                "bessel_order_pool requires gaussian_bessel or mixed beam_type"
            )
        mode_combinations = [
            tuple(mode_combination)
            for subset_size in range(
                1,
                len(bessel_order_pool) + 1,
            )
            for mode_combination in combinations(
                bessel_order_pool,
                subset_size,
            )
        ]
    else:
        mode_combinations = [tuple(bessel_orders)]

    for local_path_id in range(n_paths):

        path_id = path_start + local_path_id

        selected_orders = mode_combinations[
            path_id % len(mode_combinations)
        ]

        trajectory = generate_turpy_trajectory(
            simulator=simulator,
            params=params,
            n_z=n_z,
            total_distance=total_distance,
            r0_min=r0_min,
            r0_max=r0_max,
            cn2=cn2,
            seed=seed + path_id,
            centered=centered,
            zero_padding=zero_padding,
            padding_factor=padding_factor,
            beam_type=beam_type,
            bessel_orders=selected_orders,
            bessel_kr=bessel_kr,
            beam_waist_range=beam_waist_range,
            initial_phase=initial_phase,
            normalize_power=normalize_power,
            fixed_mode_coefficients=fixed_mode_coefficients,
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
                "bessel_orders": selected_orders,
                "mode_combination_index": path_id % len(mode_combinations),
                "realization_index": path_id // len(mode_combinations),
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
        "dx": float(params["dx"]),
        "transverse_window": float(params["dx"]) * W,
        "wavelength": float(params["wavelength"]),
        "n0": float(params["n"]),
        "outer_scale": float(params["L0"]),
        "inner_scale": float(params["l0"]),
        "cn2": cn2,
        "path_start": path_start,
        "beam_type": beam_type,
        "bessel_orders": tuple(bessel_orders),
        "bessel_kr": bessel_kr,
        "beam_waist_range": tuple(beam_waist_range),
        "initial_phase": initial_phase,
        "normalize_power": normalize_power,
        "fixed_mode_coefficients": fixed_mode_coefficients,
        "bessel_order_pool": (
            tuple(bessel_order_pool)
            if bessel_order_pool is not None
            else None
        ),
        "mode_combination_count": len(mode_combinations),
        "input_schema": "rho0 + delta_n_or_mask_slots + history_fraction",
        "future_mask_value": 0.0,
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
    dx=0.03125,
    subharmonics=True,
    subharmonic_levels=3,
    wavelength=655e-9,
    n0=1.00027,
    outer_scale=30.0,
    inner_scale=5e-3,
    n_paths=100,
    n_z=21,
    total_distance=2000.0,
    r0_min=0.03,
    r0_max=0.15,
    cn2=1e-15,
    seed=123,
    path_start=0,
    centered=True,
    zero_padding=True,
    padding_factor=2,
    beam_type="gaussian_bessel",
    bessel_orders=(0,),
    bessel_kr=20.0,
    bessel_order_pool=None,
    beam_waist_range=(0.5, 0.5),
    initial_phase="flat",
    normalize_power=True,
    fixed_mode_coefficients=True,
):
    """Generate and save one independent HPC chunk."""

    from turpy import make_turpy_simulator

    params, simulator = make_turpy_simulator(
        grid_size=grid_size,
        dx=dx,
        subharmonics=subharmonics,
        subharmonic_levels=subharmonic_levels,
        wavelength=wavelength,
        n0=n0,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )

    dataset = make_one_step_dataset(
        simulator=simulator,
        params=params,
        n_paths=n_paths,
        n_z=n_z,
        total_distance=total_distance,
        r0_min=r0_min,
        r0_max=r0_max,
        cn2=cn2,
        seed=seed,
        path_start=path_start,
        centered=centered,
        zero_padding=zero_padding,
        padding_factor=padding_factor,
        beam_type=beam_type,
        bessel_orders=bessel_orders,
        bessel_kr=bessel_kr,
        bessel_order_pool=bessel_order_pool,
        beam_waist_range=beam_waist_range,
        initial_phase=initial_phase,
        normalize_power=normalize_power,
        fixed_mode_coefficients=fixed_mode_coefficients,
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
        for beam_key in (
            "dx",
            "transverse_window",
            "wavelength",
            "n0",
            "outer_scale",
            "inner_scale",
            "cn2",
            "beam_type",
            "bessel_orders",
            "bessel_kr",
            "bessel_order_pool",
            "mode_combination_count",
            "beam_waist_range",
            "initial_phase",
            "normalize_power",
            "fixed_mode_coefficients",
        ):
            if chunk.get(beam_key) != reference.get(beam_key):
                raise ValueError(
                    f"{beam_key} mismatch in {path}: "
                    f"{chunk.get(beam_key)} != "
                    f"{reference.get(beam_key)}"
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
        "dx": reference.get("dx"),
        "transverse_window": reference.get("transverse_window"),
        "wavelength": reference.get("wavelength"),
        "n0": reference.get("n0"),
        "outer_scale": reference.get("outer_scale"),
        "inner_scale": reference.get("inner_scale"),
        "cn2": reference.get("cn2"),
        "beam_type": reference.get("beam_type", "gaussian"),
        "bessel_orders": reference.get("bessel_orders", (0,)),
        "bessel_kr": reference.get("bessel_kr", 20.0),
        "bessel_order_pool": reference.get(
            "bessel_order_pool",
            None,
        ),
        "mode_combination_count": reference.get(
            "mode_combination_count",
            1,
        ),
        "beam_waist_range": reference.get("beam_waist_range"),
        "initial_phase": reference.get("initial_phase"),
        "normalize_power": reference.get("normalize_power"),
        "fixed_mode_coefficients": reference.get(
            "fixed_mode_coefficients"
        ),
        "input_schema": reference.get(
            "input_schema",
            "rho0 + delta_n_or_mask_slots + history_fraction",
        ),
        "future_mask_value": reference.get(
            "future_mask_value",
            0.0,
        ),
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
    parser.add_argument(
        "--dx", type=float, default=0.03125,
        help="Transverse spacing in meters; 0.03125 gives a 2 m window at 64x64.",
    )
    parser.add_argument("--total-distance", type=float, default=2000.0)
    parser.add_argument("--r0-min", type=float, default=0.03)
    parser.add_argument("--r0-max", type=float, default=0.15)
    parser.add_argument("--cn2", type=float, default=1e-15)
    parser.add_argument(
        "--random-r0", action="store_true",
        help="Ignore --cn2 and draw each slab r0 from --r0-min/--r0-max.",
    )
    parser.add_argument("--wavelength", type=float, default=655e-9)
    parser.add_argument("--n0", type=float, default=1.00027)
    parser.add_argument("--outer-scale", type=float, default=30.0)
    parser.add_argument("--inner-scale", type=float, default=5e-3)
    parser.add_argument("--subharmonic-levels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument(
        "--no-subharmonics",
        action="store_true",
    )
    parser.add_argument(
        "--centered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use zero beam offset and zero input tilt.",
    )
    parser.add_argument(
        "--zero-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero-pad each propagation step before FFT propagation.",
    )
    parser.add_argument(
        "--padding-factor",
        type=int,
        default=2,
        help="Padding grid size multiplier; default is 2.",
    )
    parser.add_argument(
        "--beam-type",
        choices=("gaussian", "gaussian_bessel", "mixed"),
        default="gaussian_bessel",
    )
    parser.add_argument(
        "--bessel-orders",
        default="0",
        help="Comma-separated integer Bessel orders, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--bessel-kr",
        type=float,
        default=20.0,
        help="Bessel radial spatial frequency in 1/m.",
    )
    parser.add_argument(
        "--bessel-order-pool",
        default="-8,-4,-2,0,3,5,7",
        help=(
            "Comma-separated order pool. All nonempty subsets are "
            "assigned cyclically across global path IDs."
        ),
    )
    parser.add_argument("--beam-waist-min", type=float, default=0.5)
    parser.add_argument("--beam-waist-max", type=float, default=0.5)
    parser.add_argument(
        "--initial-phase", choices=("flat", "vortex"), default="flat"
    )
    parser.add_argument(
        "--normalize-power",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fixed-mode-coefficients",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use equal coefficients so a mode subset defines one repeatable IC.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bessel_orders = tuple(
        int(order.strip())
        for order in args.bessel_orders.split(",")
        if order.strip()
    )
    bessel_order_pool = None
    if args.bessel_order_pool is not None:
        bessel_order_pool = tuple(
            int(order.strip())
            for order in args.bessel_order_pool.split(",")
            if order.strip()
        )

    if args.mode == "generate":
        if args.beam_waist_min <= 0 or args.beam_waist_max < args.beam_waist_min:
            raise ValueError("Beam-waist limits must be positive and ascending")
        if args.cn2 <= 0 and not args.random_r0:
            raise ValueError("--cn2 must be positive unless --random-r0 is used")
        save_dataset_chunk(
            args.output,
            grid_size=args.grid_size,
            dx=args.dx,
            subharmonics=not args.no_subharmonics,
            subharmonic_levels=args.subharmonic_levels,
            wavelength=args.wavelength,
            n0=args.n0,
            outer_scale=args.outer_scale,
            inner_scale=args.inner_scale,
            n_paths=args.n_paths,
            n_z=args.n_z,
            total_distance=args.total_distance,
            r0_min=args.r0_min,
            r0_max=args.r0_max,
            cn2=None if args.random_r0 else args.cn2,
            seed=args.seed,
            path_start=args.path_start,
            centered=args.centered,
            zero_padding=args.zero_padding,
            padding_factor=args.padding_factor,
            beam_type=args.beam_type,
            bessel_orders=bessel_orders,
            bessel_kr=args.bessel_kr,
            bessel_order_pool=bessel_order_pool,
            beam_waist_range=(args.beam_waist_min, args.beam_waist_max),
            initial_phase=args.initial_phase,
            normalize_power=args.normalize_power,
            fixed_mode_coefficients=args.fixed_mode_coefficients,
        )
    else:
        merge_dataset_chunks(
            args.chunk_dir,
            args.output,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
        )
