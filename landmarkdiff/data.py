"""Reusable data loading utilities for LandmarkDiff training and evaluation.

Provides PyTorch Dataset implementations for loading synthetic training pairs,
manifest-based datasets, and evaluation datasets. Extracted from the training
script for reuse across training, evaluation, and testing pipelines.

Usage::

    from landmarkdiff.data import SurgicalPairDataset, create_dataloader

    dataset = SurgicalPairDataset("data/training_combined", resolution=512)
    loader = create_dataloader(dataset, batch_size=4, num_workers=4)

    for batch in loader:
        input_img = batch["input"]       # (B, 3, H, W) RGB [0,1]
        target_img = batch["target"]     # (B, 3, H, W) RGB [0,1]
        conditioning = batch["conditioning"]  # (B, 3, H, W) RGB [0,1]
        mask = batch["mask"]             # (B, 1, H, W) [0,1]
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------


class SurgicalPairDataset(Dataset):
    """Dataset for loading surgical before/after training pairs.

    Each sample has four components:
    - input: original face image (before surgery)
    - target: modified face image (after surgery)
    - conditioning: 3-channel landmark mesh visualization
    - mask: surgical region mask (soft float)

    Supports loading from a flat directory of ``{prefix}_input.png`` files
    or from a manifest CSV.

    Args:
        data_dir: Directory containing training pair images.
        resolution: Target image resolution (square).
        manifest_path: Optional CSV with columns [prefix, procedure, ...].
            If None, auto-discovers pairs from ``*_input.png`` files.
        transform: Optional callable for custom augmentation. Receives and
            returns a dict with numpy arrays.
    """

    def __init__(
        self,
        data_dir: str | Path,
        resolution: int = 512,
        manifest_path: str | Path | None = None,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.transform = transform

        # Discover pairs
        if manifest_path is not None:
            self.pairs, self.metadata = self._load_manifest(Path(manifest_path))
        else:
            self.pairs = sorted(self.data_dir.glob("*_input.png"))
            self.metadata = self._load_metadata()

        if not self.pairs:
            raise FileNotFoundError(f"No training pairs found in {data_dir}")

        logger.info("Loaded %d training pairs from %s", len(self.pairs), data_dir)

    def _load_manifest(self, path: Path) -> tuple[list[Path], dict[str, dict]]:
        """Load pairs from a manifest CSV."""
        pairs = []
        metadata = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                prefix = row.get("prefix", row.get("name", ""))
                input_path = self.data_dir / f"{prefix}_input.png"
                if input_path.exists():
                    pairs.append(input_path)
                    metadata[prefix] = dict(row)
        return pairs, metadata

    def _load_metadata(self) -> dict[str, dict]:
        """Load metadata from metadata.json if present."""
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path) as f:
                data = json.load(f)
            result: dict = data.get("pairs", {})
            return result
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to load metadata from %s", meta_path)
            return {}

    def get_procedure(self, idx: int) -> str:
        """Get the surgical procedure type for a sample."""
        prefix = self._prefix(idx)
        info = self.metadata.get(prefix, {})
        proc: str = info.get("procedure", "unknown")
        return proc

    def get_procedures(self) -> list[str]:
        """Get procedure types for all samples."""
        return [self.get_procedure(i) for i in range(len(self))]

    def _prefix(self, idx: int) -> str:
        return self.pairs[idx].stem.replace("_input", "")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        prefix = self._prefix(idx)

        # Load images as BGR uint8
        input_bgr = self._load_image(f"{prefix}_input.png")
        target_bgr = self._load_image(f"{prefix}_target.png")
        cond_bgr = self._load_image(f"{prefix}_conditioning.png")
        mask_arr = self._load_mask(f"{prefix}_mask.png")

        sample = {
            "input_image": input_bgr,
            "target_image": target_bgr,
            "conditioning": cond_bgr,
            "mask": mask_arr,
            "procedure": self.get_procedure(idx),
            "idx": idx,
        }

        # Apply custom transform
        if self.transform is not None:
            sample = self.transform(sample)

        # Convert to tensors
        return {
            "input": bgr_to_tensor(sample["input_image"]),
            "target": bgr_to_tensor(sample["target_image"]),
            "conditioning": bgr_to_tensor(sample["conditioning"]),
            "mask": mask_to_tensor(sample["mask"]),
            "procedure": sample["procedure"],
            "idx": sample["idx"],
        }

    def _load_image(self, filename: str) -> np.ndarray:
        """Load an image as BGR uint8, resized to resolution."""
        path = self.data_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Failed to load %s, using blank", path)
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        if img.shape[:2] != (self.resolution, self.resolution):
            img = cv2.resize(img, (self.resolution, self.resolution))
        return img

    def _load_mask(self, filename: str) -> np.ndarray:
        """Load a mask as float32 [0,1], resized to resolution."""
        path = self.data_dir / filename
        if not path.exists():
            return np.ones((self.resolution, self.resolution), dtype=np.float32)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.ones((self.resolution, self.resolution), dtype=np.float32)
        mask = cv2.resize(mask, (self.resolution, self.resolution))
        return mask.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Evaluation dataset (input + ground truth)
# ---------------------------------------------------------------------------


class EvalPairDataset(Dataset):
    """Dataset for evaluation: loads input/target pairs with procedure labels.

    Args:
        data_dir: Directory with evaluation pairs.
        resolution: Target resolution.
    """

    def __init__(self, data_dir: str | Path, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.pairs = sorted(self.data_dir.glob("*_input.png"))

        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        self._meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    self._meta = json.load(f).get("pairs", {})
            except (json.JSONDecodeError, OSError):
                logger.debug("Failed to load metadata from %s", meta_path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        prefix = self.pairs[idx].stem.replace("_input", "")

        input_img = self._load(f"{prefix}_input.png")
        target_img = self._load(f"{prefix}_target.png")

        info = self._meta.get(prefix, {})
        procedure = info.get("procedure", "unknown")

        return {
            "input": bgr_to_tensor(input_img),
            "target": bgr_to_tensor(target_img),
            "procedure": procedure,
            "prefix": prefix,
        }

    def _load(self, filename: str) -> np.ndarray:
        path = self.data_dir / filename
        img = cv2.imread(str(path))
        if img is None:
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        if img.shape[:2] != (self.resolution, self.resolution):
            img = cv2.resize(img, (self.resolution, self.resolution))
        return img


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 image to RGB [0,1] tensor (C, H, W)."""
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert RGB [0,1] tensor (C, H, W) to BGR uint8 image."""
    rgb = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    bgr = (rgb[:, :, ::-1] * 255).astype(np.uint8)
    return np.ascontiguousarray(bgr)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert float32 mask (H, W) to tensor (1, H, W)."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return torch.from_numpy(mask).unsqueeze(0)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def create_procedure_sampler(
    dataset: SurgicalPairDataset,
    balance_procedures: bool = True,
) -> Sampler | None:
    """Create a weighted sampler that balances procedure types.

    Returns None if balancing is disabled or all procedures are the same.
    """
    if not balance_procedures:
        return None

    procedures = dataset.get_procedures()
    unique_procs = list(set(procedures))

    if len(unique_procs) <= 1:
        return None

    # Count per procedure
    counts = {p: procedures.count(p) for p in unique_procs}
    total = len(procedures)

    # Weight inversely proportional to count
    weights = []
    for proc in procedures:
        w = total / (len(unique_procs) * counts[proc])
        weights.append(w)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    sampler: Sampler | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Shuffle data (ignored if sampler is provided).
        sampler: Custom sampler (e.g., from create_procedure_sampler).
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop last incomplete batch.
        persistent_workers: Keep workers alive between epochs.

    Returns:
        Configured DataLoader.
    """
    if sampler is not None:
        shuffle = False  # Sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Multi-directory dataset
# ---------------------------------------------------------------------------


class CombinedDataset(Dataset):
    """Combine multiple SurgicalPairDatasets into one.

    Useful for combining synthetic v1, v2, v3 data and real pairs.

    Args:
        datasets: List of SurgicalPairDataset instances.
    """

    def __init__(self, datasets: list[SurgicalPairDataset]):
        self.datasets = datasets
        self._cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self._cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"CombinedDataset index {idx} out of range [0, {len(self)})")
        dataset_idx = 0
        for i, size in enumerate(self._cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        if dataset_idx > 0:
            idx -= self._cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][idx]

    def get_procedure(self, idx: int) -> str:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"CombinedDataset index {idx} out of range [0, {len(self)})")
        dataset_idx = 0
        for i, size in enumerate(self._cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        if dataset_idx > 0:
            idx -= self._cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_procedure(idx)

    def get_procedures(self) -> list[str]:
        procs = []
        for ds in self.datasets:
            procs.extend(ds.get_procedures())
        return procs
