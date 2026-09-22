"""Chunked TurPy datasets and a two-dimensional Fourier Neural Operator."""

from __future__ import annotations

import bisect
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, TensorDataset


def load_turpy_file(path: Path) -> dict:
    """Load a trusted TurPy data file on the CPU across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # ``weights_only`` is unavailable in older PyTorch.
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class ChunkInfo:
    """Small in-memory index for one otherwise lazily loaded chunk."""

    path: Path
    sample_count: int
    sample_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    path_ids: tuple[int, ...]
    mode_combination_ids: tuple[int, ...]
    n_z: int
    total_distance: float


@dataclass
class Normalization:
    """Training-only scalar normalization values."""

    intensity_mean: float
    intensity_std: float
    delta_n_rms: float

    def state_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, values: dict[str, float]) -> "Normalization":
        return cls(**values)

    def normalize_target(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.intensity_mean) / self.intensity_std

    def denormalize_target(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.intensity_std + self.intensity_mean


def scan_turpy_chunks(data_dir: Path, pattern: str = "*.pt") -> list[ChunkInfo]:
    """Validate chunks while retaining only their small path-ID indices."""
    data_dir = Path(data_dir)
    paths = sorted(path for path in data_dir.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {data_dir}")

    infos: list[ChunkInfo] = []
    reference: tuple[tuple[int, ...], tuple[int, ...], int, float] | None = None
    path_owner: dict[int, Path] = {}
    for path in paths:
        chunk = load_turpy_file(path)
        missing = {"X", "Y", "path_ids", "n_z", "total_distance"} - chunk.keys()
        if missing:
            raise ValueError(f"{path} is not a TurPy chunk; missing keys: {sorted(missing)}")
        x, y, path_ids = chunk["X"], chunk["Y"], chunk["path_ids"]
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"Expected rank-4 X/Y tensors in {path}; got {x.shape} and {y.shape}")
        if y.shape[-1] != 1:
            raise ValueError(f"Expected one target channel in {path}; got {y.shape[-1]}")
        if x.shape[0] != y.shape[0] or x.shape[0] != path_ids.numel():
            raise ValueError(f"X, Y, and path_ids sample counts disagree in {path}")
        if x.shape[-1] != int(chunk["n_z"]) + 1:
            raise ValueError(
                f"Expected n_z + 1 input channels in {path}; "
                f"got {x.shape[-1]} channels for n_z={chunk['n_z']}"
            )
        signature = (
            tuple(x.shape[1:]), tuple(y.shape[1:]),
            int(chunk["n_z"]), float(chunk["total_distance"]),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"Chunk shape or propagation metadata mismatch in {path}")

        integer_ids = tuple(int(value) for value in path_ids.tolist())
        ordered_path_ids = list(dict.fromkeys(integer_ids))
        path_metadata = chunk.get("path_metadata", [])
        mode_combination_count = chunk.get("mode_combination_count")
        if path_metadata:
            if len(path_metadata) != len(ordered_path_ids):
                raise ValueError(
                    f"path_metadata count does not match unique paths in {path}"
                )
            path_to_mode = {}
            for path_id, metadata in zip(ordered_path_ids, path_metadata):
                if "mode_combination_index" in metadata:
                    mode_id = int(metadata["mode_combination_index"])
                elif mode_combination_count is not None:
                    mode_id = path_id % int(mode_combination_count)
                else:
                    mode_id = path_id
                path_to_mode[path_id] = mode_id
        elif mode_combination_count is not None:
            path_to_mode = {
                path_id: path_id % int(mode_combination_count)
                for path_id in ordered_path_ids
            }
        else:
            # Older one-realization chunks have no IC metadata. Treating each
            # path as its own IC preserves their original split behavior.
            path_to_mode = {path_id: path_id for path_id in ordered_path_ids}
        mode_ids = tuple(path_to_mode[path_id] for path_id in integer_ids)
        for path_id in set(integer_ids):
            if path_id in path_owner:
                raise ValueError(
                    f"Path ID {path_id} occurs in both {path_owner[path_id]} and {path}. "
                    "Use non-overlapping --path-start values."
                )
            path_owner[path_id] = path
        infos.append(ChunkInfo(
            path=path,
            sample_count=x.shape[0],
            sample_shape=tuple(x.shape[1:]),
            target_shape=tuple(y.shape[1:]),
            path_ids=integer_ids,
            mode_combination_ids=mode_ids,
            n_z=int(chunk["n_z"]),
            total_distance=float(chunk["total_distance"]),
        ))
        del chunk, x, y, path_ids
    return infos


def split_path_ids(
    chunks: Sequence[ChunkInfo],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 123,
    split_unit: str = "mode-combination",
) -> dict[str, list[int]]:
    """Split complete paths, optionally grouping all realizations of an IC."""
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("Validation and test fractions must be between zero and one")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than one")
    if split_unit not in ("mode-combination", "path"):
        raise ValueError("split_unit must be 'mode-combination' or 'path'")

    path_to_mode: dict[int, int] = {}
    for chunk in chunks:
        for path_id, mode_id in zip(chunk.path_ids, chunk.mode_combination_ids):
            previous = path_to_mode.setdefault(path_id, mode_id)
            if previous != mode_id:
                raise ValueError(f"Path {path_id} maps to multiple mode combinations")

    path_to_group = {
        path_id: mode_id if split_unit == "mode-combination" else path_id
        for path_id, mode_id in path_to_mode.items()
    }
    unique_groups = sorted(set(path_to_group.values()))
    if len(unique_groups) < 3:
        raise ValueError(
            f"At least three {split_unit} groups are required for train/val/test splits"
        )
    random.Random(seed).shuffle(unique_groups)
    n_test = max(1, round(len(unique_groups) * test_fraction))
    n_val = max(1, round(len(unique_groups) * val_fraction))
    if n_test + n_val >= len(unique_groups):
        raise ValueError("The requested split leaves no training paths")
    group_splits = {
        "test": set(unique_groups[:n_test]),
        "val": set(unique_groups[n_test:n_test + n_val]),
        "train": set(unique_groups[n_test + n_val:]),
    }
    path_splits = {
        name: sorted(
            path_id
            for path_id, group_id in path_to_group.items()
            if group_id in selected_groups
        )
        for name, selected_groups in group_splits.items()
    }
    if split_unit == "mode-combination":
        mode_sets = [
            {path_to_mode[path_id] for path_id in path_splits[name]}
            for name in ("train", "val", "test")
        ]
        if any(mode_sets[i] & mode_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("Mode-combination leakage detected across splits")
    return path_splits


def mode_combination_ids_for_paths(
    chunks: Sequence[ChunkInfo],
    path_ids: Iterable[int],
) -> list[int]:
    """Return sorted unique IC IDs represented by a collection of paths."""
    selected_paths = set(path_ids)
    selected_modes: set[int] = set()
    for chunk in chunks:
        for path_id, mode_id in zip(chunk.path_ids, chunk.mode_combination_ids):
            if path_id in selected_paths:
                selected_modes.add(mode_id)
    return sorted(selected_modes)


def _local_indices(chunk: ChunkInfo, selected_path_ids: set[int]) -> torch.Tensor:
    return torch.tensor(
        [i for i, path_id in enumerate(chunk.path_ids) if path_id in selected_path_ids],
        dtype=torch.long,
    )


def compute_chunked_normalization(
    chunks: Sequence[ChunkInfo],
    train_path_ids: Iterable[int],
    scan_batch_size: int = 16,
) -> Normalization:
    """Compute training-only moments while loading one chunk at a time.

    The delta-n RMS excludes zero-filled future slots. Dividing by an RMS
    without subtracting a mean keeps unavailable slots exactly zero.
    """
    train_ids = set(train_path_ids)
    if not train_ids or scan_batch_size < 1:
        raise ValueError("Training IDs and scan_batch_size must be nonempty/positive")
    intensity_sum = intensity_sq_sum = delta_sq_sum = 0.0
    intensity_count = delta_count = 0
    for info in chunks:
        indices = _local_indices(info, train_ids)
        if indices.numel() == 0:
            continue
        chunk = load_turpy_file(info.path)
        n_slots = info.n_z - 1
        for start in range(0, indices.numel(), scan_batch_size):
            selection = indices[start:start + scan_batch_size]
            x = chunk["X"].index_select(0, selection)
            y = chunk["Y"].index_select(0, selection)
            rho0, target = x[..., 0], y[..., 0]
            intensity_sum += rho0.sum(dtype=torch.float64).item()
            intensity_sum += target.sum(dtype=torch.float64).item()
            intensity_sq_sum += rho0.square().sum(dtype=torch.float64).item()
            intensity_sq_sum += target.square().sum(dtype=torch.float64).item()
            intensity_count += rho0.numel() + target.numel()
            active_counts = torch.round(x[:, 0, 0, -1] * n_slots).long()
            for slot in range(n_slots):
                active = active_counts > slot
                if active.any():
                    values = x[active, ..., 1 + slot]
                    delta_sq_sum += values.square().sum(dtype=torch.float64).item()
                    delta_count += values.numel()
            del x, y
        del chunk
    if intensity_count == 0 or delta_count == 0:
        raise ValueError("Training chunks contain no usable values")
    intensity_mean = intensity_sum / intensity_count
    variance = max(
        intensity_sq_sum / intensity_count - intensity_mean**2,
        1.0e-20,
    )
    return Normalization(
        intensity_mean=intensity_mean,
        intensity_std=variance**0.5,
        delta_n_rms=max((delta_sq_sum / delta_count) ** 0.5, 1.0e-20),
    )


class ChunkedTurpyDataset(Dataset):
    """Dataset view that keeps at most one complete ``.pt`` chunk cached."""

    def __init__(
        self,
        chunks: Sequence[ChunkInfo],
        selected_path_ids: Iterable[int],
        normalization: Normalization,
    ) -> None:
        selected_ids = set(selected_path_ids)
        self.normalization = normalization
        self.entries: list[tuple[ChunkInfo, torch.Tensor]] = []
        self.chunk_ranges: list[tuple[int, int]] = []
        self._ends: list[int] = []
        total = 0
        for info in chunks:
            indices = _local_indices(info, selected_ids)
            if indices.numel() == 0:
                continue
            self.entries.append((info, indices))
            self.chunk_ranges.append((total, total + indices.numel()))
            total += indices.numel()
            self._ends.append(total)
        if total == 0:
            raise ValueError("No samples matched the selected path IDs")
        self._length = total
        self._cached_entry = -1
        self._cached_chunk: dict | None = None

    def __len__(self) -> int:
        return self._length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cached_entry"] = -1
        state["_cached_chunk"] = None
        return state

    def clear_cache(self) -> None:
        """Release the currently cached chunk before changing data phases."""
        self._cached_chunk = None
        self._cached_entry = -1

    def _load_entry(self, entry_index: int) -> dict:
        if entry_index != self._cached_entry:
            # Drop the old reference before loading the next large file. This
            # avoids briefly holding two complete chunks during assignment.
            self.clear_cache()
            self._cached_chunk = load_turpy_file(self.entries[entry_index][0].path)
            self._cached_entry = entry_index
        assert self._cached_chunk is not None
        return self._cached_chunk

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        entry_index = bisect.bisect_right(self._ends, index)
        previous_end = 0 if entry_index == 0 else self._ends[entry_index - 1]
        _, indices = self.entries[entry_index]
        local_index = int(indices[index - previous_end])
        chunk = self._load_entry(entry_index)
        x = chunk["X"][local_index].float().clone()
        y = chunk["Y"][local_index].float().clone()
        path_id = chunk["path_ids"][local_index].long().clone()
        x[..., 0] = (
            x[..., 0] - self.normalization.intensity_mean
        ) / self.normalization.intensity_std
        x[..., 1:-1] = x[..., 1:-1] / self.normalization.delta_n_rms
        y = self.normalization.normalize_target(y)
        return {"x": x, "y": y, "path_id": path_id}


class ChunkShuffleSampler(Sampler):
    """Shuffle chunks and samples without repeatedly reopening large chunks."""

    def __init__(self, dataset: ChunkedTurpyDataset, seed: int = 123) -> None:
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        ranges = list(self.dataset.chunk_ranges)
        rng.shuffle(ranges)
        for start, end in ranges:
            indices = list(range(start, end))
            rng.shuffle(indices)
            yield from indices


class SpectralConv2d(nn.Module):
    """Low-mode complex Fourier convolution over two spatial axes."""

    def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        if modes_y < 1 or modes_x < 1:
            raise ValueError("Fourier mode counts must be positive")
        self.modes_y, self.modes_x = modes_y, modes_x
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, modes_y, modes_x)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _contract(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("biyx,ioyx->boyx", values, weights)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = value.shape
        if height < 2 or width < 2:
            raise ValueError("SpectralConv2d requires spatial dimensions of at least 2")
        transformed = torch.fft.rfft2(value, dim=(-2, -1))
        output = torch.zeros(
            batch, self.weight_positive.shape[1], height, width // 2 + 1,
            dtype=transformed.dtype, device=value.device,
        )
        my = min(self.modes_y, height // 2)
        mx = min(self.modes_x, width // 2 + 1)
        output[:, :, :my, :mx] = self._contract(
            transformed[:, :, :my, :mx], self.weight_positive[:, :, :my, :mx]
        )
        output[:, :, -my:, :mx] = self._contract(
            transformed[:, :, -my:, :mx], self.weight_negative[:, :, :my, :mx]
        )
        return torch.fft.irfft2(output, s=(height, width), dim=(-2, -1))


class FNO2d(nn.Module):
    """Map TurPy history ``[B,H,W,C]`` to next intensity ``[B,H,W,1]``."""

    def __init__(
        self,
        input_channels: int = 22,
        modes_y: int = 16,
        modes_x: int = 16,
        width: int = 32,
        layers: int = 4,
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        if input_channels < 3 or width < 1 or layers < 1:
            raise ValueError("input_channels, width, and layers are invalid")
        self.input_channels, self.use_skip = input_channels, use_skip
        self.lift = nn.Conv2d(input_channels + 2, width, 1)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes_y, modes_x) for _ in range(layers)
        ])
        self.local_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, 1, 1)
        )

    @staticmethod
    def _coordinates(value: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = value.shape
        y = torch.linspace(-1.0, 1.0, height, dtype=value.dtype, device=value.device)
        x = torch.linspace(-1.0, 1.0, width, dtype=value.dtype, device=value.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx)).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"Expected rank-4 input, got {tuple(value.shape)}")
        if value.shape[-1] == self.input_channels:
            value = value.permute(0, 3, 1, 2)
        elif value.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels in first or last axis; "
                f"got {tuple(value.shape)}"
            )
        value = self.lift(torch.cat((value, self._coordinates(value)), dim=1))
        for spectral, local in zip(self.spectral_layers, self.local_layers):
            update = spectral(value) + local(value)
            value = F.gelu(update + value if self.use_skip else update)
        return self.project(value).permute(0, 2, 3, 1)


def get_turpy_dataloaders(dataset: dict, batch_size: int = 16):
    """Backward-compatible loaders for an already merged dataset."""
    base = TensorDataset(dataset["X"], dataset["Y"])
    splits = [Subset(base, dataset["splits"][f"{name}_idx"]) for name in ("train", "val", "test")]
    return (
        DataLoader(splits[0], batch_size=batch_size, shuffle=True),
        DataLoader(splits[1], batch_size=batch_size, shuffle=False),
        DataLoader(splits[2], batch_size=batch_size, shuffle=False),
    )
