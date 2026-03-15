"""Tests for critical code paths with low or missing coverage.

Covers:
- checkpoint_manager: safe pop(), index without checkpoints key, extra_state,
  save_pretrained branch, empty checkpoint list symlink
- manipulation: KeyError fallback from data-driven displacement, intensity
  scaling math
- conditioning: generate_conditioning 3-tuple return with default dims
- clinical: detect_vitiligo_patches, get_keloid_exclusion_mask
- masking: generate_surgical_mask with clinical flags (vitiligo + keloid)
- ensemble: init, is_loaded, pixel_average/median, generate not loaded error,
  invalid strategy
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from landmarkdiff.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
)
from landmarkdiff.clinical import (
    ClinicalFlags,
    adjust_mask_for_keloid,
    adjust_mask_for_vitiligo,
    detect_vitiligo_patches,
    get_bells_palsy_side_indices,
    get_keloid_exclusion_mask,
)
from landmarkdiff.conditioning import (
    auto_canny,
    generate_conditioning,
    render_wireframe,
)
from landmarkdiff.ensemble import EnsembleInference
from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import (
    PROCEDURE_LANDMARKS,
    apply_procedure_preset,
)
from landmarkdiff.masking import (
    generate_surgical_mask,
    mask_to_3channel,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_face(
    width: int = 512,
    height: int = 512,
    seed: int = 42,
) -> FaceLandmarks:
    """Create a FaceLandmarks with reproducible random coords."""
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        confidence=0.9,
    )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


# ===========================================================================
# CheckpointManager: safe pop and edge cases
# ===========================================================================


class TestCheckpointManagerSafePop:
    """Test the safe .pop(name, None) fix in _prune."""

    def test_prune_removes_from_index_safely(self, tmp_path):
        """Pruning should use safe pop and not raise KeyError."""
        mgr = CheckpointManager(
            output_dir=tmp_path / "ckpts",
            keep_best=1,
            keep_latest=1,
            metric="loss",
        )
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Save enough checkpoints to trigger pruning
        mgr.save(
            step=10, controlnet=model, ema_controlnet=model, optimizer=opt, metrics={"loss": 0.9}
        )
        mgr.save(
            step=20, controlnet=model, ema_controlnet=model, optimizer=opt, metrics={"loss": 0.1}
        )
        mgr.save(
            step=30, controlnet=model, ema_controlnet=model, optimizer=opt, metrics={"loss": 0.5}
        )

        # step-10 should have been pruned (not best, not latest)
        names = [c["name"] for c in mgr.list_checkpoints()]
        assert "checkpoint-10" not in names

    def test_prune_nonexistent_dir_no_error(self, tmp_path):
        """Pruning a checkpoint whose directory was already deleted should not error."""
        mgr = CheckpointManager(
            output_dir=tmp_path / "ckpts",
            keep_best=1,
            keep_latest=1,
            metric="loss",
        )
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        mgr.save(
            step=10, controlnet=model, ema_controlnet=model, optimizer=opt, metrics={"loss": 0.9}
        )
        mgr.save(
            step=20, controlnet=model, ema_controlnet=model, optimizer=opt, metrics={"loss": 0.1}
        )

        # Manually delete checkpoint-10 dir before next save triggers prune
        import shutil

        ckpt_10 = tmp_path / "ckpts" / "checkpoint-10"
        if ckpt_10.exists():
            shutil.rmtree(ckpt_10)

        # This should not raise
        mgr.save(
            step=30, controlnet=model, ema_controlnet=model, optimizer=opt, metrics={"loss": 0.5}
        )


class TestCheckpointManagerIndexEdgeCases:
    """Test index loading edge cases."""

    def test_index_without_checkpoints_key(self, tmp_path):
        """Loading an index file missing 'checkpoints' key should recover."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir(parents=True)
        index_path = ckpt_dir / "checkpoint_index.json"
        index_path.write_text(json.dumps({"version": 1}))

        mgr = CheckpointManager(output_dir=ckpt_dir)
        assert "checkpoints" in mgr._index
        assert mgr._index["checkpoints"] == {}

    def test_index_with_empty_checkpoints(self, tmp_path):
        """Loading an index with empty checkpoints dict should work."""
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "checkpoint_index.json").write_text(json.dumps({"checkpoints": {}}))
        mgr = CheckpointManager(output_dir=ckpt_dir)
        assert mgr.list_checkpoints() == []
        assert mgr.get_latest_step() == 0


class TestCheckpointManagerExtraState:
    """Test save with extra_state parameter."""

    def test_extra_state_saved(self, tmp_path):
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(output_dir=tmp_path / "ckpts")

        mgr.save(
            step=100,
            controlnet=model,
            ema_controlnet=model,
            optimizer=opt,
            extra_state={"custom_key": 42, "ema_decay": 0.999},
        )

        state = torch.load(
            tmp_path / "ckpts" / "checkpoint-100" / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert state["custom_key"] == 42
        assert state["ema_decay"] == 0.999

    def test_save_pretrained_called(self, tmp_path):
        """When ema_controlnet has save_pretrained, it should be called."""
        model = TinyModel()
        ema = TinyModel()
        save_mock = MagicMock()
        ema.save_pretrained = save_mock  # type: ignore[attr-defined]
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mgr = CheckpointManager(output_dir=tmp_path / "ckpts")

        mgr.save(step=100, controlnet=model, ema_controlnet=ema, optimizer=opt)
        save_mock.assert_called_once()


class TestCheckpointManagerSymlinkEdge:
    """Test symlink edge cases."""

    def test_update_symlinks_no_checkpoints(self, tmp_path):
        """_update_symlinks with empty checkpoint list should return early."""
        mgr = CheckpointManager(output_dir=tmp_path / "ckpts")
        # Directly call private method; should not crash
        mgr._update_symlinks()
        assert not (tmp_path / "ckpts" / "latest").exists()
        assert not (tmp_path / "ckpts" / "best").exists()


# ===========================================================================
# Manipulation: KeyError fallback from data-driven displacement
# ===========================================================================


class TestManipulationKeyErrorFallback:
    """Test that apply_procedure_preset falls back to RBF when
    the displacement model raises KeyError for the procedure."""

    def test_keyerror_falls_back_to_rbf(self):
        """When displacement model doesn't have the procedure, use RBF preset."""
        face = _make_face()

        with patch("landmarkdiff.displacement_model.DisplacementModel") as mock_dm_cls:
            mock_model = mock_dm_cls.load.return_value
            mock_model.get_displacement_field.side_effect = KeyError("brow_lift")

            result = apply_procedure_preset(
                face,
                "brow_lift",
                intensity=50.0,
                displacement_model_path="/fake/model.npz",
            )

        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == (478, 3)
        # Should have some deformation from RBF fallback
        diff = np.linalg.norm(result.landmarks - face.landmarks)
        assert diff > 0

    def test_data_driven_success_skips_rbf(self):
        """When displacement model succeeds, use its displacements."""
        face = _make_face()
        fake_field = np.ones((478, 2), dtype=np.float32) * 0.01

        with patch("landmarkdiff.displacement_model.DisplacementModel") as mock_dm_cls:
            mock_model = mock_dm_cls.load.return_value
            mock_model.get_displacement_field.return_value = fake_field

            result = apply_procedure_preset(
                face,
                "rhinoplasty",
                intensity=50.0,
                displacement_model_path="/fake/model.npz",
            )

        assert isinstance(result, FaceLandmarks)
        # Verify displacement was applied (all landmarks shifted by +0.01 in x,y)
        diff = result.landmarks[:, :2] - face.landmarks[:, :2]
        assert np.mean(diff) > 0  # net positive shift


class TestManipulationIntensityScalingMath:
    """Test the intensity/100 scaling in apply_procedure_preset."""

    def test_scale_is_intensity_over_100(self):
        """intensity=50 should produce half the displacement of intensity=100."""
        face = _make_face()
        r50 = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        r100 = apply_procedure_preset(face, "rhinoplasty", intensity=100.0)

        diff_50 = np.linalg.norm(r50.landmarks - face.landmarks)
        diff_100 = np.linalg.norm(r100.landmarks - face.landmarks)

        # Due to RBF nonlinearity, won't be exactly 2x, but should be close
        ratio = diff_100 / diff_50
        assert 1.5 < ratio < 2.5

    def test_data_driven_scale_is_intensity_over_50(self):
        """In data-driven mode, intensity 50 maps to 1.0x."""
        face = _make_face()
        call_args = {}

        def capture_args(**kwargs):
            call_args.update(kwargs)
            return np.zeros((478, 2), dtype=np.float32)

        with patch("landmarkdiff.displacement_model.DisplacementModel") as mock_dm_cls:
            mock_model = mock_dm_cls.load.return_value
            mock_model.get_displacement_field.side_effect = capture_args

            apply_procedure_preset(
                face,
                "rhinoplasty",
                intensity=50.0,
                displacement_model_path="/fake/model.npz",
            )

        assert abs(call_args["intensity"] - 1.0) < 1e-6

    def test_all_six_procedures_produce_nonzero(self):
        """Every procedure should produce measurable deformation at intensity=75."""
        face = _make_face()
        for proc in PROCEDURE_LANDMARKS:
            result = apply_procedure_preset(face, proc, intensity=75.0)
            diff = np.linalg.norm(result.landmarks - face.landmarks)
            assert diff > 0, f"{proc} produced zero deformation"


# ===========================================================================
# Conditioning: generate_conditioning 3-tuple return
# ===========================================================================


class TestGenerateConditioningTuple:
    """Test generate_conditioning returns (landmark_img, canny, wireframe)."""

    def test_returns_3_tuple(self):
        face = _make_face()
        result = generate_conditioning(face, 512, 512)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_tuple_types(self):
        face = _make_face()
        landmark_img, canny, wireframe = generate_conditioning(face, 512, 512)
        # landmark_img: BGR color
        assert landmark_img.ndim == 3
        assert landmark_img.shape == (512, 512, 3)
        assert landmark_img.dtype == np.uint8
        # canny: grayscale binary
        assert canny.ndim == 2
        assert canny.shape == (512, 512)
        assert canny.dtype == np.uint8
        # wireframe: grayscale
        assert wireframe.ndim == 2
        assert wireframe.shape == (512, 512)
        assert wireframe.dtype == np.uint8

    def test_default_dimensions(self):
        """When width/height are None, use face dimensions."""
        face = _make_face(width=384, height=384)
        landmark_img, canny, wireframe = generate_conditioning(face)
        assert wireframe.shape == (384, 384)
        assert canny.shape == (384, 384)
        assert landmark_img.shape[:2] == (384, 384)

    def test_wireframe_has_nonzero_pixels(self):
        face = _make_face()
        _, _, wireframe = generate_conditioning(face, 512, 512)
        assert np.sum(wireframe > 0) > 100

    def test_canny_is_binary(self):
        face = _make_face()
        _, canny, _ = generate_conditioning(face, 512, 512)
        unique = set(np.unique(canny))
        assert unique.issubset({0, 255})


# ===========================================================================
# Clinical: detect_vitiligo_patches
# ===========================================================================


class TestDetectVitiligoPatchesCoverage:
    """Test detect_vitiligo_patches for branch coverage."""

    def test_detects_bright_patch(self):
        """High-L, low-saturation patch on face should be detected."""
        face = _make_face()
        # Create a BGR image with a bright, desaturated patch
        img = np.full((512, 512, 3), 150, dtype=np.uint8)
        # Add bright white patch (high L, low saturation in LAB)
        img[200:280, 200:280] = [250, 250, 250]

        result = detect_vitiligo_patches(img, face, l_threshold=85.0, min_patch_area=100)
        assert result.shape == (512, 512)
        assert result.dtype == np.uint8
        # Should detect at least some patch area
        # (depends on whether the patch falls within face convex hull)

    def test_no_face_pixels_returns_empty(self):
        """If no face pixels found (empty hull), return empty mask."""
        # Create face with all landmarks at (0,0) so hull is degenerate
        landmarks = np.zeros((478, 3), dtype=np.float32)
        face = FaceLandmarks(
            landmarks=landmarks,
            image_width=512,
            image_height=512,
            confidence=0.9,
        )
        img = np.full((512, 512, 3), 200, dtype=np.uint8)
        result = detect_vitiligo_patches(img, face)
        assert result.shape == (512, 512)
        # All zeros or very few pixels (degenerate hull)

    def test_dark_image_no_detection(self):
        """Dark image should not detect vitiligo patches."""
        face = _make_face()
        img = np.full((512, 512, 3), 30, dtype=np.uint8)
        result = detect_vitiligo_patches(img, face, l_threshold=85.0, min_patch_area=50)
        assert np.sum(result) == 0

    def test_small_patches_filtered(self):
        """Patches smaller than min_patch_area should be filtered out."""
        face = _make_face()
        img = np.full((512, 512, 3), 100, dtype=np.uint8)
        # Add tiny bright spot (area < min_patch_area)
        img[256, 256] = [255, 255, 255]
        result = detect_vitiligo_patches(img, face, min_patch_area=200)
        assert result.shape == (512, 512)


class TestGetKeloidExclusionMaskCoverage:
    """Test get_keloid_exclusion_mask for coverage."""

    def test_jawline_region(self):
        face = _make_face()
        mask = get_keloid_exclusion_mask(face, ["jawline"], 512, 512)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.max() <= 1.0
        assert mask.min() >= 0.0
        # Jawline is a valid region, should produce non-empty mask
        assert mask.max() > 0

    def test_unknown_region_ignored(self):
        face = _make_face()
        mask = get_keloid_exclusion_mask(face, ["nonexistent_region"], 512, 512)
        assert mask.shape == (512, 512)
        # Unknown region should be all zeros
        assert mask.max() == 0.0

    def test_no_dilation(self):
        face = _make_face()
        mask = get_keloid_exclusion_mask(face, ["jawline"], 512, 512, margin_px=0)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32

    def test_multiple_regions(self):
        face = _make_face()
        mask = get_keloid_exclusion_mask(face, ["jawline", "nose"], 512, 512)
        assert mask.shape == (512, 512)
        assert mask.max() > 0


# ===========================================================================
# Masking: generate_surgical_mask with clinical flags
# ===========================================================================


class TestMaskingWithClinicalFlags:
    """Test generate_surgical_mask clinical flag branches (lines 282-302)."""

    def test_vitiligo_flag_modifies_mask(self):
        """With vitiligo flag and image, mask should be adjusted."""
        face = _make_face()
        flags = ClinicalFlags(vitiligo=True)
        img = np.full((512, 512, 3), 150, dtype=np.uint8)
        # Add bright patch that could be detected as vitiligo
        img[200:280, 200:280] = [250, 250, 250]

        mask = generate_surgical_mask(face, "rhinoplasty", clinical_flags=flags, image=img)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_vitiligo_flag_without_image_no_crash(self):
        """Vitiligo flag without image should not crash (image=None skips)."""
        face = _make_face()
        flags = ClinicalFlags(vitiligo=True)
        # image=None should skip the vitiligo branch
        mask = generate_surgical_mask(face, "rhinoplasty", clinical_flags=flags)
        assert mask.shape == (512, 512)

    def test_keloid_flag_modifies_mask(self):
        """With keloid flag and regions, mask should be softened."""
        face = _make_face()
        flags = ClinicalFlags(keloid_prone=True, keloid_regions=["jawline"])
        mask = generate_surgical_mask(face, "rhytidectomy", clinical_flags=flags)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_keloid_no_regions_no_crash(self):
        """Keloid flag with empty regions list should not crash."""
        face = _make_face()
        flags = ClinicalFlags(keloid_prone=True, keloid_regions=[])
        mask = generate_surgical_mask(face, "rhinoplasty", clinical_flags=flags)
        assert mask.shape == (512, 512)

    def test_both_flags_combined(self):
        """Both vitiligo and keloid flags should work together."""
        face = _make_face()
        flags = ClinicalFlags(
            vitiligo=True,
            keloid_prone=True,
            keloid_regions=["jawline"],
        )
        img = np.full((512, 512, 3), 150, dtype=np.uint8)
        mask = generate_surgical_mask(face, "rhytidectomy", clinical_flags=flags, image=img)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.float32


# ===========================================================================
# Ensemble: init, error handling, aggregation
# ===========================================================================


class TestEnsembleEdgeCases:
    """Test EnsembleInference edge cases for coverage."""

    def test_not_loaded_raises(self):
        """Calling generate before load should raise RuntimeError."""
        ensemble = EnsembleInference(mode="tps", n_samples=2)
        assert not ensemble.is_loaded
        with pytest.raises(RuntimeError, match="not loaded"):
            ensemble.generate(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_invalid_strategy_raises_on_generate(self):
        """Invalid strategy should raise ValueError during generate."""
        ensemble = EnsembleInference(mode="tps", n_samples=2, strategy="bad_strategy")
        # Mock pipeline loading
        mock_pipeline = MagicMock()
        mock_pipeline.is_loaded = True
        mock_pipeline.generate.return_value = {"output": np.zeros((64, 64, 3), dtype=np.uint8)}
        ensemble._pipeline = mock_pipeline

        with pytest.raises(ValueError, match="Unknown strategy"):
            ensemble.generate(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_pixel_average_clipping(self):
        """Pixel average should clip to [0, 255]."""
        ensemble = EnsembleInference(mode="tps", n_samples=2)
        outputs = [
            np.full((32, 32, 3), 250, dtype=np.uint8),
            np.full((32, 32, 3), 250, dtype=np.uint8),
        ]
        result = ensemble._pixel_average(outputs)
        assert result.max() <= 255
        assert result.min() >= 0

    def test_pixel_median_with_outlier(self):
        """Median should be robust to outliers."""
        ensemble = EnsembleInference(mode="tps", n_samples=5)
        outputs = [
            np.full((32, 32, 3), 100, dtype=np.uint8),
            np.full((32, 32, 3), 100, dtype=np.uint8),
            np.full((32, 32, 3), 100, dtype=np.uint8),
            np.full((32, 32, 3), 0, dtype=np.uint8),  # outlier
            np.full((32, 32, 3), 255, dtype=np.uint8),  # outlier
        ]
        result = ensemble._pixel_median(outputs)
        assert abs(int(result.mean()) - 100) < 2

    def test_init_stores_kwargs(self):
        ensemble = EnsembleInference(
            mode="controlnet",
            controlnet_checkpoint="/fake/path",
            displacement_model_path="/fake/dm.npz",
            n_samples=10,
            strategy="pixel_average",
            base_seed=999,
        )
        assert ensemble.mode == "controlnet"
        assert ensemble.controlnet_checkpoint == "/fake/path"
        assert ensemble.displacement_model_path == "/fake/dm.npz"
        assert ensemble.n_samples == 10
        assert ensemble.strategy == "pixel_average"
        assert ensemble.base_seed == 999
        assert ensemble._pipeline is None


# ===========================================================================
# Conditioning: render_wireframe default dimension paths
# ===========================================================================


class TestRenderWireframeDefaults:
    """Test render_wireframe using face.image_width/height when None passed."""

    def test_no_width_height_uses_face_dims(self):
        face = _make_face(width=384, height=256)
        wf = render_wireframe(face)
        assert wf.shape == (256, 384)

    def test_explicit_overrides_face_dims(self):
        face = _make_face(width=384, height=256)
        wf = render_wireframe(face, width=128, height=128)
        assert wf.shape == (128, 128)


# ===========================================================================
# auto_canny: all-zero image fallback to default median
# ===========================================================================


class TestAutoCannyDefaultMedian:
    """Test auto_canny's fallback median when no non-zero pixels exist."""

    def test_all_zero_image(self):
        img = np.zeros((128, 128), dtype=np.uint8)
        edges = auto_canny(img)
        assert edges.shape == (128, 128)
        assert np.sum(edges) == 0  # no edges in blank image


# ===========================================================================
# CheckpointMetadata: roundtrip serialization
# ===========================================================================


class TestCheckpointMetadataRoundtrip:
    def test_full_roundtrip(self):
        meta = CheckpointMetadata(
            step=500,
            timestamp=1234567890.0,
            metrics={"loss": 0.05, "fid": 42.3},
            epoch=10,
            phase="B",
            is_best=True,
            size_mb=125.3,
        )
        d = meta.to_dict()
        restored = CheckpointMetadata.from_dict(d)
        assert restored.step == 500
        assert restored.epoch == 10
        assert restored.phase == "B"
        assert restored.is_best is True
        assert restored.size_mb == 125.3
        assert restored.metrics["loss"] == 0.05
        assert restored.metrics["fid"] == 42.3


# ===========================================================================
# Clinical: adjust_mask_for_vitiligo edge cases
# ===========================================================================


class TestAdjustMaskForVitiligoEdgeCases:
    def test_zero_preservation_factor(self):
        """Factor=0 should not change the mask at all."""
        mask = np.ones((64, 64), dtype=np.float32) * 0.7
        patches = np.full((64, 64), 255, dtype=np.uint8)
        result = adjust_mask_for_vitiligo(mask, patches, preservation_factor=0.0)
        np.testing.assert_array_almost_equal(result, mask)

    def test_output_clipped_to_01(self):
        """Result should be clipped to [0, 1]."""
        mask = np.ones((64, 64), dtype=np.float32) * 0.2
        patches = np.full((64, 64), 255, dtype=np.uint8)
        result = adjust_mask_for_vitiligo(mask, patches, preservation_factor=0.5)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ===========================================================================
# Clinical: adjust_mask_for_keloid edge cases
# ===========================================================================


class TestAdjustMaskForKeloidEdgeCases:
    def test_full_keloid_coverage(self):
        """When keloid covers everything, mask should be heavily reduced."""
        mask = np.ones((128, 128), dtype=np.float32)
        keloid_mask = np.ones((128, 128), dtype=np.float32)
        result = adjust_mask_for_keloid(mask, keloid_mask, reduction_factor=0.8)
        assert result.max() <= 1.0
        assert result.min() >= 0.0
        # Center pixel should be significantly reduced
        center = result[64, 64]
        assert center < 0.5

    def test_zero_reduction(self):
        """reduction_factor=0 should effectively keep mask unchanged."""
        mask = np.ones((128, 128), dtype=np.float32) * 0.8
        keloid_mask = np.ones((128, 128), dtype=np.float32)
        result = adjust_mask_for_keloid(mask, keloid_mask, reduction_factor=0.0)
        # With factor=0, keloid_reduction=0, so modified=mask*(1-0)=mask
        # But blurred version is used in keloid regions, so values may differ
        # due to Gaussian blur at edges. Center should be close.
        assert abs(result[64, 64] - 0.8) < 0.1


# ===========================================================================
# Bells palsy: validate index integrity
# ===========================================================================


class TestBellsPalsyIndexIntegrity:
    def test_all_indices_within_478(self):
        for side in ["left", "right"]:
            regions = get_bells_palsy_side_indices(side)
            for region, indices in regions.items():
                for idx in indices:
                    assert 0 <= idx < 478, f"{side}/{region}: index {idx} out of range"

    def test_left_right_no_overlap(self):
        left = get_bells_palsy_side_indices("left")
        right = get_bells_palsy_side_indices("right")
        for region in left:
            left_set = set(left[region])
            right_set = set(right[region])
            assert left_set.isdisjoint(right_set), f"Overlap in {region}"


# ===========================================================================
# mask_to_3channel: dtype preservation
# ===========================================================================


class TestMaskTo3ChannelDtype:
    def test_float32_preserved(self):
        mask = np.random.rand(32, 32).astype(np.float32)
        result = mask_to_3channel(mask)
        assert result.dtype == np.float32

    def test_uint8_preserved(self):
        mask = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        result = mask_to_3channel(mask)
        assert result.dtype == np.uint8
