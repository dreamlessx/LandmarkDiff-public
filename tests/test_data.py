"""Tests for the data loading module."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from landmarkdiff.data import (
    CombinedDataset,
    EvalPairDataset,
    SurgicalPairDataset,
    bgr_to_tensor,
    create_dataloader,
    create_procedure_sampler,
    mask_to_tensor,
    tensor_to_bgr,
)


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temp directory with sample training pairs."""
    for i in range(6):
        # Vary procedure across pairs
        proc = ["rhinoplasty", "blepharoplasty", "rhytidectomy"][i % 3]

        # Create input, target, conditioning, mask images
        for suffix in ["input", "target", "conditioning"]:
            img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"{proc}_{i:06d}_{suffix}.png"), img)

        mask = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{proc}_{i:06d}_mask.png"), mask)

    # Create metadata.json
    metadata = {"pairs": {}}
    for i in range(6):
        proc = ["rhinoplasty", "blepharoplasty", "rhytidectomy"][i % 3]
        prefix = f"{proc}_{i:06d}"
        metadata["pairs"][prefix] = {"procedure": proc}

    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path


@pytest.fixture
def sample_manifest(tmp_path, sample_data_dir):
    """Create a manifest CSV for the sample data."""
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix", "procedure"])
        writer.writeheader()
        for i in range(6):
            proc = ["rhinoplasty", "blepharoplasty", "rhytidectomy"][i % 3]
            writer.writerow({"prefix": f"{proc}_{i:06d}", "procedure": proc})
    return manifest


class TestBgrToTensor:
    """Tests for BGR to tensor conversion."""

    def test_shape(self):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t = bgr_to_tensor(bgr)
        assert t.shape == (3, 64, 64)

    def test_range(self):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t = bgr_to_tensor(bgr)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_dtype(self):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t = bgr_to_tensor(bgr)
        assert t.dtype == torch.float32

    def test_channel_order_swap(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 100  # Blue channel
        bgr[:, :, 2] = 200  # Red channel
        t = bgr_to_tensor(bgr)
        # After BGR->RGB swap, channel 0 should be Red (200/255)
        assert abs(t[0].mean().item() - 200 / 255) < 0.01
        # Channel 2 should be Blue (100/255)
        assert abs(t[2].mean().item() - 100 / 255) < 0.01


class TestTensorToBgr:
    """Tests for tensor to BGR conversion."""

    def test_roundtrip(self):
        bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        t = bgr_to_tensor(bgr)
        recovered = tensor_to_bgr(t)
        np.testing.assert_array_equal(recovered, bgr)

    def test_shape(self):
        t = torch.randn(3, 32, 32).clamp(0, 1)
        bgr = tensor_to_bgr(t)
        assert bgr.shape == (32, 32, 3)
        assert bgr.dtype == np.uint8


class TestMaskToTensor:
    """Tests for mask conversion."""

    def test_2d_input(self):
        mask = np.random.rand(64, 64).astype(np.float32)
        t = mask_to_tensor(mask)
        assert t.shape == (1, 64, 64)

    def test_3d_input(self):
        mask = np.random.rand(64, 64, 3).astype(np.float32)
        t = mask_to_tensor(mask)
        assert t.shape == (1, 64, 64)


class TestSurgicalPairDataset:
    """Tests for the main training dataset."""

    def test_length(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        assert len(ds) == 6

    def test_item_keys(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        item = ds[0]
        assert "input" in item
        assert "target" in item
        assert "conditioning" in item
        assert "mask" in item
        assert "procedure" in item
        assert "idx" in item

    def test_item_shapes(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        item = ds[0]
        assert item["input"].shape == (3, 64, 64)
        assert item["target"].shape == (3, 64, 64)
        assert item["conditioning"].shape == (3, 64, 64)
        assert item["mask"].shape == (1, 64, 64)

    def test_item_ranges(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        item = ds[0]
        for key in ["input", "target", "conditioning"]:
            assert item[key].min() >= 0.0
            assert item[key].max() <= 1.0
        assert item["mask"].min() >= 0.0
        assert item["mask"].max() <= 1.0

    def test_get_procedure(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        procs = ds.get_procedures()
        assert len(procs) == 6
        assert set(procs) == {"rhinoplasty", "blepharoplasty", "rhytidectomy"}

    def test_manifest_loading(self, sample_data_dir, sample_manifest):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64, manifest_path=sample_manifest)
        assert len(ds) == 6

    def test_custom_resolution(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=128)
        item = ds[0]
        assert item["input"].shape == (3, 128, 128)

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SurgicalPairDataset(tmp_path)

    def test_missing_mask_defaults_to_ones(self, tmp_path):
        # Create a pair without mask
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for suffix in ["input", "target", "conditioning"]:
            cv2.imwrite(str(tmp_path / f"test_{suffix}.png"), img)

        ds = SurgicalPairDataset(tmp_path, resolution=64)
        item = ds[0]
        assert item["mask"].min() == 1.0
        assert item["mask"].max() == 1.0

    def test_custom_transform(self, sample_data_dir):
        def flip_transform(sample):
            sample["input_image"] = np.flip(sample["input_image"], axis=1).copy()
            sample["target_image"] = np.flip(sample["target_image"], axis=1).copy()
            return sample

        ds = SurgicalPairDataset(sample_data_dir, resolution=64, transform=flip_transform)
        item = ds[0]
        assert item["input"].shape == (3, 64, 64)


class TestEvalPairDataset:
    """Tests for the evaluation dataset."""

    def test_length(self, sample_data_dir):
        ds = EvalPairDataset(sample_data_dir, resolution=64)
        assert len(ds) == 6

    def test_item_keys(self, sample_data_dir):
        ds = EvalPairDataset(sample_data_dir, resolution=64)
        item = ds[0]
        assert "input" in item
        assert "target" in item
        assert "procedure" in item
        assert "prefix" in item

    def test_item_shapes(self, sample_data_dir):
        ds = EvalPairDataset(sample_data_dir, resolution=64)
        item = ds[0]
        assert item["input"].shape == (3, 64, 64)
        assert item["target"].shape == (3, 64, 64)


class TestCombinedDataset:
    """Tests for combining multiple datasets."""

    def test_combined_length(self, sample_data_dir, tmp_path):
        ds1 = SurgicalPairDataset(sample_data_dir, resolution=64)

        # Create a second small dataset
        dir2 = tmp_path / "ds2"
        dir2.mkdir()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for suffix in ["input", "target", "conditioning"]:
            cv2.imwrite(str(dir2 / f"extra_{suffix}.png"), img)

        ds2 = SurgicalPairDataset(dir2, resolution=64)
        combined = CombinedDataset([ds1, ds2])
        assert len(combined) == 7

    def test_combined_access(self, sample_data_dir, tmp_path):
        ds1 = SurgicalPairDataset(sample_data_dir, resolution=64)

        dir2 = tmp_path / "ds2"
        dir2.mkdir()
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        for suffix in ["input", "target", "conditioning"]:
            cv2.imwrite(str(dir2 / f"extra_{suffix}.png"), img)

        ds2 = SurgicalPairDataset(dir2, resolution=64)
        combined = CombinedDataset([ds1, ds2])

        # Access first dataset items
        item0 = combined[0]
        assert item0["input"].shape == (3, 64, 64)

        # Access second dataset item
        item_last = combined[6]
        assert item_last["input"].shape == (3, 64, 64)


class TestCreateProcedureSampler:
    """Tests for procedure-balanced sampling."""

    def test_returns_sampler(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        sampler = create_procedure_sampler(ds, balance_procedures=True)
        assert sampler is not None
        assert len(list(sampler)) == len(ds)

    def test_returns_none_if_disabled(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        sampler = create_procedure_sampler(ds, balance_procedures=False)
        assert sampler is None

    def test_returns_none_single_procedure(self, tmp_path):
        # All same procedure
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            for suffix in ["input", "target", "conditioning"]:
                cv2.imwrite(str(tmp_path / f"test_{i}_{suffix}.png"), img)

        ds = SurgicalPairDataset(tmp_path, resolution=64)
        sampler = create_procedure_sampler(ds, balance_procedures=True)
        assert sampler is None  # All "unknown", single procedure


class TestCreateDataloader:
    """Tests for DataLoader creation."""

    def test_creates_loader(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        loader = create_dataloader(ds, batch_size=2, num_workers=0)
        assert isinstance(loader, DataLoader)

    def test_batch_shape(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        loader = create_dataloader(ds, batch_size=2, num_workers=0, drop_last=False)
        batch = next(iter(loader))
        assert batch["input"].shape == (2, 3, 64, 64)

    def test_default_keeps_last_batch(self, sample_data_dir):
        """Default drop_last=False should not discard the final partial batch."""
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        n = len(ds)
        # Use a batch size that doesn't divide evenly
        bs = max(2, n - 1) if n > 2 else 2
        loader = create_dataloader(ds, batch_size=bs, num_workers=0, shuffle=False)
        total = sum(batch["input"].shape[0] for batch in loader)
        assert total == n

    def test_with_sampler(self, sample_data_dir):
        ds = SurgicalPairDataset(sample_data_dir, resolution=64)
        sampler = create_procedure_sampler(ds)
        loader = create_dataloader(ds, batch_size=2, num_workers=0, sampler=sampler)
        batch = next(iter(loader))
        assert batch["input"].shape[0] == 2
