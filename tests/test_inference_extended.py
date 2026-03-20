"""Extended tests for inference module -- utility functions that don't need models."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from landmarkdiff.landmarks import FaceLandmarks, TPSLandmarkResult  # noqa: E402


def _make_face(seed=42, width=512, height=512):
    """Create a mock FaceLandmarks."""
    rng = np.random.default_rng(seed)
    lm = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        lm[i, 0] = 0.3 + rng.random() * 0.4
        lm[i, 1] = 0.2 + rng.random() * 0.6
        lm[i, 2] = rng.random() * 0.1
    return FaceLandmarks(landmarks=lm, confidence=0.95, image_width=width, image_height=height)


class TestGetDevice:
    """Tests for get_device."""

    def test_returns_torch_device(self):
        from landmarkdiff.inference import get_device

        device = get_device()
        assert isinstance(device, torch.device)


class TestNumpyToPil:
    """Tests for numpy_to_pil."""

    def test_bgr_image(self):
        from landmarkdiff.inference import numpy_to_pil

        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        pil = numpy_to_pil(arr)
        assert isinstance(pil, Image.Image)
        assert pil.size == (64, 64)

    def test_grayscale_image(self):
        from landmarkdiff.inference import numpy_to_pil

        arr = np.full((64, 64), 128, dtype=np.uint8)
        pil = numpy_to_pil(arr)
        assert isinstance(pil, Image.Image)
        assert pil.mode == "L"


class TestPilToNumpy:
    """Tests for pil_to_numpy."""

    def test_rgb_image(self):
        from landmarkdiff.inference import pil_to_numpy

        pil = Image.new("RGB", (64, 64), (100, 150, 200))
        arr = pil_to_numpy(pil)
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8

    def test_grayscale_image(self):
        from landmarkdiff.inference import pil_to_numpy

        pil = Image.new("L", (64, 64), 128)
        arr = pil_to_numpy(pil)
        assert arr.shape == (64, 64)


class TestLoadImage:
    """Tests for load_image."""

    def test_load_png(self, tmp_path):
        import cv2

        from landmarkdiff.inference import load_image

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        result = load_image(str(path))
        assert result is not None
        assert result.shape == (64, 64, 3)

    def test_load_nonexistent(self, tmp_path):
        from landmarkdiff.inference import load_image

        result = load_image(str(tmp_path / "nonexistent.png"))
        assert result is None

    def test_load_jpg(self, tmp_path):
        import cv2

        from landmarkdiff.inference import load_image

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        result = load_image(str(path))
        assert result is not None


class TestCheckImageQuality:
    """Tests for check_image_quality."""

    def test_good_image_no_warnings(self):
        from landmarkdiff.inference import check_image_quality

        rng = np.random.default_rng(42)
        img = rng.integers(50, 200, (512, 512, 3), dtype=np.uint8)
        warnings = check_image_quality(img)
        assert isinstance(warnings, list)

    def test_low_resolution_warning(self):
        from landmarkdiff.inference import check_image_quality

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        warnings = check_image_quality(img)
        assert any("resolution" in w.lower() or "Low" in w for w in warnings)

    def test_dark_image_warning(self):
        from landmarkdiff.inference import check_image_quality

        img = np.full((256, 256, 3), 10, dtype=np.uint8)
        warnings = check_image_quality(img)
        assert any("dark" in w.lower() or "Dark" in w for w in warnings)

    def test_blurry_image_warning(self):
        from landmarkdiff.inference import check_image_quality

        # Uniform color = zero Laplacian variance = blurry
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        warnings = check_image_quality(img)
        assert any("blur" in w.lower() or "Blur" in w for w in warnings)


class TestMaskComposite:
    """Tests for mask_composite."""

    def test_basic_composite(self):
        from landmarkdiff.inference import mask_composite

        warped = np.full((64, 64, 3), 200, dtype=np.uint8)
        original = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = mask_composite(warped, original, mask)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_full_mask(self):
        from landmarkdiff.inference import mask_composite

        warped = np.full((64, 64, 3), 200, dtype=np.uint8)
        original = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = mask_composite(warped, original, mask)
        assert result.shape == (64, 64, 3)

    def test_uint8_mask(self):
        from landmarkdiff.inference import mask_composite

        warped = np.full((64, 64, 3), 200, dtype=np.uint8)
        original = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        result = mask_composite(warped, original, mask)
        assert result.shape == (64, 64, 3)


class TestEstimateFaceView:
    """Tests for estimate_face_view."""

    def test_returns_view_info(self):
        from landmarkdiff.inference import estimate_face_view

        face = _make_face()
        info = estimate_face_view(face)
        assert "yaw" in info
        assert "pitch" in info
        assert "view" in info
        assert "is_frontal" in info
        assert info["view"] in ("frontal", "three_quarter", "profile")

    def test_is_frontal_bool(self):
        from landmarkdiff.inference import estimate_face_view

        face = _make_face()
        info = estimate_face_view(face)
        assert isinstance(info["is_frontal"], bool)


class TestGetOptimalDtype:
    """Tests for get_optimal_dtype."""

    def test_returns_dtype(self):
        from landmarkdiff.inference import get_optimal_dtype

        dtype = get_optimal_dtype()
        assert dtype in (torch.float16, torch.float32)

    def test_cpu_returns_float32(self):
        from landmarkdiff.inference import get_optimal_dtype

        dtype = get_optimal_dtype(device=torch.device("cpu"))
        assert dtype == torch.float32


class TestEstimateVramUsage:
    """Tests for estimate_vram_usage."""

    def test_returns_dict(self):
        from landmarkdiff.inference import estimate_vram_usage

        result = estimate_vram_usage()
        assert "total_gb" in result
        assert "unet_gb" in result
        assert "controlnet_gb" in result
        assert result["total_gb"] > 0

    def test_tps_mode_no_controlnet(self):
        from landmarkdiff.inference import estimate_vram_usage

        result = estimate_vram_usage(mode="tps")
        assert result["controlnet_gb"] == 0.0

    def test_higher_resolution_more_vram(self):
        from landmarkdiff.inference import estimate_vram_usage

        r512 = estimate_vram_usage(resolution=512)
        r1024 = estimate_vram_usage(resolution=1024)
        assert r1024["activations_gb"] >= r512["activations_gb"]


class TestMatchSkinTone:
    """Tests for _match_skin_tone."""

    def test_basic_match(self):
        from landmarkdiff.inference import _match_skin_tone

        source = np.full((64, 64, 3), 120, dtype=np.uint8)
        target = np.full((64, 64, 3), 160, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = _match_skin_tone(source, target, mask)
        assert result.shape == (64, 64, 3)

    def test_no_mask(self):
        from landmarkdiff.inference import _match_skin_tone

        source = np.full((64, 64, 3), 120, dtype=np.uint8)
        target = np.full((64, 64, 3), 160, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = _match_skin_tone(source, target, mask)
        # With no mask, should return source unchanged
        np.testing.assert_array_equal(result, source)


class TestProcedurePrompts:
    """Tests for PROCEDURE_PROMPTS and NEGATIVE_PROMPT."""

    def test_procedure_prompts_exist(self):
        from landmarkdiff.inference import PROCEDURE_PROMPTS

        assert "rhinoplasty" in PROCEDURE_PROMPTS
        assert "blepharoplasty" in PROCEDURE_PROMPTS
        assert "rhytidectomy" in PROCEDURE_PROMPTS
        assert "orthognathic" in PROCEDURE_PROMPTS
        assert "brow_lift" in PROCEDURE_PROMPTS
        assert "mentoplasty" in PROCEDURE_PROMPTS

    def test_negative_prompt(self):
        from landmarkdiff.inference import NEGATIVE_PROMPT

        assert isinstance(NEGATIVE_PROMPT, str)
        assert len(NEGATIVE_PROMPT) > 0


class TestLandmarkDiffPipeline:
    """Tests for LandmarkDiffPipeline init and properties."""

    def test_tps_mode_is_loaded(self):
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipe = LandmarkDiffPipeline(mode="tps")
        assert pipe.is_loaded is True

    def test_not_loaded_by_default(self):
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipe = LandmarkDiffPipeline(mode="img2img")
        assert pipe.is_loaded is False

    def test_tps_load_noop(self):
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipe = LandmarkDiffPipeline(mode="tps")
        pipe.load()  # Should not raise
        assert pipe.is_loaded is True

    def test_invalid_tps_backend_raises(self):
        from landmarkdiff.inference import LandmarkDiffPipeline

        with pytest.raises(ValueError, match="tps_backend"):
            LandmarkDiffPipeline(mode="tps", tps_backend="invalid")

    def test_generate_uses_onnx_tps_backend_when_enabled(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)
        face = _make_face(width=512, height=512)
        expected = np.full((512, 512, 3), 201, dtype=np.uint8)
        calls = {"init_path": None, "init_calls": 0, "warp_calls": 0}

        class DummyRuntime:
            def __init__(self, onnx_path):
                calls["init_calls"] += 1
                calls["init_path"] = onnx_path

            def warp(self, img, src, dst):
                calls["warp_calls"] += 1
                assert img.shape == (512, 512, 3)
                assert src.shape[1] == 2
                assert dst.shape[1] == 2
                return expected

        monkeypatch.setattr("landmarkdiff.inference.extract_landmarks", lambda _img: face)
        monkeypatch.setattr(
            "landmarkdiff.inference.apply_procedure_preset",
            lambda *args, **kwargs: face,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.render_landmark_image",
            lambda *args, **kwargs: np.zeros((512, 512, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.generate_surgical_mask",
            lambda *args, **kwargs: np.ones((512, 512), dtype=np.float32),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.mask_composite",
            lambda warped, original, mask: warped,
        )
        monkeypatch.setattr("landmarkdiff.inference.TPSONNXRuntime", DummyRuntime)
        monkeypatch.setattr(
            "landmarkdiff.inference.warp_image_tps",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("fallback should not run")
            ),
        )

        pipe = LandmarkDiffPipeline(mode="tps", tps_backend="onnx", tps_onnx_path="dummy.onnx")
        result = pipe.generate(image, procedure="rhinoplasty", postprocess=False)
        pipe.generate(image, procedure="rhinoplasty", postprocess=False)

        assert calls["init_path"] == "dummy.onnx"
        assert calls["init_calls"] == 1
        assert calls["warp_calls"] == 2
        np.testing.assert_array_equal(result["output_tps"], expected)
        np.testing.assert_array_equal(result["output"], expected)

    def test_generate_falls_back_to_numpy_when_onnx_init_fails(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)
        face = _make_face(width=512, height=512)
        fallback = np.full((512, 512, 3), 155, dtype=np.uint8)
        calls = {"init_calls": 0, "fallback_calls": 0}

        class FailingRuntime:
            def __init__(self, onnx_path):
                calls["init_calls"] += 1
                raise RuntimeError(f"cannot load {onnx_path}")

        def fake_fallback(*args, **kwargs):
            calls["fallback_calls"] += 1
            return fallback

        monkeypatch.setattr("landmarkdiff.inference.extract_landmarks", lambda _img: face)
        monkeypatch.setattr(
            "landmarkdiff.inference.apply_procedure_preset",
            lambda *args, **kwargs: face,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.render_landmark_image",
            lambda *args, **kwargs: np.zeros((512, 512, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.generate_surgical_mask",
            lambda *args, **kwargs: np.ones((512, 512), dtype=np.float32),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.mask_composite",
            lambda warped, original, mask: warped,
        )
        monkeypatch.setattr("landmarkdiff.inference.TPSONNXRuntime", FailingRuntime)
        monkeypatch.setattr("landmarkdiff.inference.warp_image_tps", fake_fallback)

        pipe = LandmarkDiffPipeline(mode="tps", tps_backend="onnx", tps_onnx_path="missing.onnx")
        result_1 = pipe.generate(image, procedure="rhinoplasty", postprocess=False)
        result_2 = pipe.generate(image, procedure="rhinoplasty", postprocess=False)

        assert calls["init_calls"] == 1
        assert calls["fallback_calls"] == 2
        np.testing.assert_array_equal(result_1["output_tps"], fallback)
        np.testing.assert_array_equal(result_2["output_tps"], fallback)

    def test_generate_falls_back_to_numpy_when_onnx_path_missing(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)
        face = _make_face(width=512, height=512)
        fallback = np.full((512, 512, 3), 166, dtype=np.uint8)
        calls = {"fallback_calls": 0}

        def fake_fallback(*args, **kwargs):
            calls["fallback_calls"] += 1
            return fallback

        monkeypatch.setattr("landmarkdiff.inference.extract_landmarks", lambda _img: face)
        monkeypatch.setattr(
            "landmarkdiff.inference.apply_procedure_preset",
            lambda *args, **kwargs: face,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.render_landmark_image",
            lambda *args, **kwargs: np.zeros((512, 512, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.generate_surgical_mask",
            lambda *args, **kwargs: np.ones((512, 512), dtype=np.float32),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.mask_composite",
            lambda warped, original, mask: warped,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.TPSONNXRuntime",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("runtime should not be initialized without path")
            ),
        )
        monkeypatch.setattr("landmarkdiff.inference.warp_image_tps", fake_fallback)

        pipe = LandmarkDiffPipeline(mode="tps", tps_backend="onnx", tps_onnx_path=None)
        result = pipe.generate(image, procedure="rhinoplasty", postprocess=False)

        assert calls["fallback_calls"] == 1
        np.testing.assert_array_equal(result["output_tps"], fallback)

    def test_generate_falls_back_when_onnx_runtime_warp_fails(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)
        face = _make_face(width=512, height=512)
        fallback = np.full((512, 512, 3), 188, dtype=np.uint8)
        calls = {"warp_calls": 0, "fallback_calls": 0}

        class RuntimeWithFailingWarp:
            def __init__(self, onnx_path):
                assert onnx_path == "dummy.onnx"

            def warp(self, img, src, dst):
                calls["warp_calls"] += 1
                raise RuntimeError("ORT execution failed")

        def fake_fallback(*args, **kwargs):
            calls["fallback_calls"] += 1
            return fallback

        monkeypatch.setattr("landmarkdiff.inference.extract_landmarks", lambda _img: face)
        monkeypatch.setattr(
            "landmarkdiff.inference.apply_procedure_preset",
            lambda *args, **kwargs: face,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.render_landmark_image",
            lambda *args, **kwargs: np.zeros((512, 512, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.generate_surgical_mask",
            lambda *args, **kwargs: np.ones((512, 512), dtype=np.float32),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.mask_composite",
            lambda warped, original, mask: warped,
        )
        monkeypatch.setattr("landmarkdiff.inference.TPSONNXRuntime", RuntimeWithFailingWarp)
        monkeypatch.setattr("landmarkdiff.inference.warp_image_tps", fake_fallback)

        pipe = LandmarkDiffPipeline(mode="tps", tps_backend="onnx", tps_onnx_path="dummy.onnx")
        result_1 = pipe.generate(image, procedure="rhinoplasty", postprocess=False)
        result_2 = pipe.generate(image, procedure="rhinoplasty", postprocess=False)

        assert calls["warp_calls"] == 1
        assert calls["fallback_calls"] == 2
        np.testing.assert_array_equal(result_1["output_tps"], fallback)
        np.testing.assert_array_equal(result_2["output_tps"], fallback)

    def test_generate_uses_tps_landmark_wrapper(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)
        face = _make_face(width=512, height=512)
        calls = {"wrapper_calls": 0}

        def fake_wrapper(
            img,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            *,
            extractor=None,
        ):
            calls["wrapper_calls"] += 1
            assert img.shape == (512, 512, 3)
            assert extractor is not None
            extracted = extractor(img, min_detection_confidence, min_tracking_confidence)
            assert extracted is face
            return TPSLandmarkResult(
                coords=face.landmarks.copy(),
                confidence=face.confidence,
                image_size=(face.image_width, face.image_height),
                detected=True,
            )

        monkeypatch.setattr("landmarkdiff.inference.extract_landmarks", lambda *_args: face)
        monkeypatch.setattr("landmarkdiff.inference.extract_tps_landmarks", fake_wrapper)
        monkeypatch.setattr(
            "landmarkdiff.inference.apply_procedure_preset",
            lambda *args, **kwargs: face,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.render_landmark_image",
            lambda *args, **kwargs: np.zeros((512, 512, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.generate_surgical_mask",
            lambda *args, **kwargs: np.ones((512, 512), dtype=np.float32),
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.mask_composite",
            lambda warped, original, mask: warped,
        )
        monkeypatch.setattr(
            "landmarkdiff.inference.warp_image_tps",
            lambda *_args, **_kwargs: np.full((512, 512, 3), 202, dtype=np.uint8),
        )

        pipe = LandmarkDiffPipeline(mode="tps")
        result = pipe.generate(image, procedure="rhinoplasty", postprocess=False)

        assert calls["wrapper_calls"] == 1
        assert result["landmarks_original"].landmarks.shape == (478, 3)
        np.testing.assert_array_equal(result["landmarks_original"].landmarks, face.landmarks)
        np.testing.assert_array_equal(result["output_tps"], np.full((512, 512, 3), 202, np.uint8))

    def test_generate_no_face_detected_raises_controlled_error(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)

        monkeypatch.setattr(
            "landmarkdiff.inference.extract_tps_landmarks",
            lambda *_args, **_kwargs: TPSLandmarkResult(
                coords=np.empty((0, 3), dtype=np.float32),
                confidence=0.0,
                image_size=(512, 512),
                detected=False,
                reason="no_face_detected",
            ),
        )

        pipe = LandmarkDiffPipeline(mode="tps")
        with pytest.raises(ValueError, match=r"No face detected in image\."):
            pipe.generate(image, procedure="rhinoplasty", postprocess=False)

    def test_generate_landmark_extractor_error_is_controlled(self, monkeypatch):
        from landmarkdiff.inference import LandmarkDiffPipeline

        image = np.full((128, 128, 3), 127, dtype=np.uint8)

        monkeypatch.setattr(
            "landmarkdiff.inference.extract_tps_landmarks",
            lambda *_args, **_kwargs: TPSLandmarkResult(
                coords=np.empty((0, 3), dtype=np.float32),
                confidence=0.0,
                image_size=(512, 512),
                detected=False,
                reason="extractor_error",
            ),
        )

        pipe = LandmarkDiffPipeline(mode="tps")
        with pytest.raises(ValueError, match="Landmark extraction failed: extractor_error"):
            pipe.generate(image, procedure="rhinoplasty", postprocess=False)
