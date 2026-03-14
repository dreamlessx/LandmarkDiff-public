"""Tests for landmarkdiff.api_client."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from landmarkdiff.api_client import LandmarkDiffClient, PredictionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image(tmp_path):
    """Create a sample image file."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)
    return path


def _encode_to_b64(img: np.ndarray) -> str:
    _, encoded = cv2.imencode(".png", img)
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# PredictionResult tests
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_save(self, tmp_path):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = PredictionResult(
            output_image=img, procedure="rhinoplasty", intensity=65.0,
        )
        out_path = tmp_path / "result.png"
        result.save(out_path)
        assert out_path.exists()
        loaded = cv2.imread(str(out_path))
        assert loaded.shape == (256, 256, 3)

    def test_defaults(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = PredictionResult(
            output_image=img, procedure="test", intensity=50.0,
        )
        assert result.confidence == 0.0
        assert result.metrics == {}
        assert result.metadata == {}
        assert result.landmarks_before is None


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_init(self):
        client = LandmarkDiffClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 60.0

    def test_trailing_slash_stripped(self):
        client = LandmarkDiffClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_repr(self):
        client = LandmarkDiffClient("http://example.com:9999")
        assert "example.com:9999" in repr(client)

    def test_context_manager(self):
        with LandmarkDiffClient("http://localhost:8000") as client:
            assert client.base_url == "http://localhost:8000"


# ---------------------------------------------------------------------------
# API calls (mocked)
# ---------------------------------------------------------------------------

class TestHealth:
    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_health(self, mock_session_fn):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok", "version": "0.3.0"}
        mock_session.get.return_value = mock_resp
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        result = client.health()
        assert result["status"] == "ok"
        mock_session.get.assert_called_once_with("http://localhost:8000/health")


class TestProcedures:
    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_procedures(self, mock_session_fn):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "procedures": ["rhinoplasty", "blepharoplasty"],
        }
        mock_session.get.return_value = mock_resp
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        procs = client.procedures()
        assert "rhinoplasty" in procs


class TestPredict:
    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_predict(self, mock_session_fn, sample_image):
        # Create mock response with base64 image
        output_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        b64_img = _encode_to_b64(output_img)

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "output_image": b64_img,
            "confidence": 0.95,
            "metrics": {"ssim": 0.87},
            "metadata": {"procedure": "rhinoplasty"},
        }
        mock_session.post.return_value = mock_resp
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        result = client.predict(
            sample_image, procedure="rhinoplasty", intensity=65.0,
        )
        assert isinstance(result, PredictionResult)
        assert result.output_image.shape == (512, 512, 3)
        assert result.confidence == 0.95
        assert result.procedure == "rhinoplasty"

    def test_predict_file_not_found(self):
        client = LandmarkDiffClient()
        with pytest.raises(FileNotFoundError):
            client.predict("/nonexistent/image.png")


class TestAnalyze:
    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_analyze(self, mock_session_fn, sample_image):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "face_detected": True,
            "fitzpatrick_type": "III",
            "landmark_count": 478,
        }
        mock_session.post.return_value = mock_resp
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        result = client.analyze(sample_image)
        assert result["face_detected"] is True
        assert result["landmark_count"] == 478


class TestBatchPredict:
    @patch("landmarkdiff.api_client.LandmarkDiffClient.predict")
    def test_batch_predict(self, mock_predict, sample_image):
        mock_predict.return_value = PredictionResult(
            output_image=np.zeros((512, 512, 3), dtype=np.uint8),
            procedure="rhinoplasty",
            intensity=65.0,
        )

        client = LandmarkDiffClient()
        results = client.batch_predict(
            [sample_image, sample_image],
            procedure="rhinoplasty",
        )
        assert len(results) == 2
        assert mock_predict.call_count == 2

    @patch("landmarkdiff.api_client.LandmarkDiffClient.predict")
    def test_batch_predict_handles_errors(self, mock_predict, sample_image):
        mock_predict.side_effect = [
            PredictionResult(
                output_image=np.zeros((512, 512, 3), dtype=np.uint8),
                procedure="rhinoplasty", intensity=65.0,
            ),
            ConnectionError("Server down"),
        ]

        client = LandmarkDiffClient()
        results = client.batch_predict(
            [sample_image, sample_image],
            procedure="rhinoplasty",
        )
        assert len(results) == 2
        # First succeeded, second has error metadata
        assert "error" in results[1].metadata


# ---------------------------------------------------------------------------
# Image encoding/decoding
# ---------------------------------------------------------------------------

class TestImageCodec:
    def test_decode_base64_image(self):
        client = LandmarkDiffClient()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        b64 = _encode_to_b64(img)
        decoded = client._decode_base64_image(b64)
        assert decoded.shape == (100, 100, 3)

    def test_decode_invalid_base64(self):
        client = LandmarkDiffClient()
        with pytest.raises(Exception):
            client._decode_base64_image("not-valid-base64!!!")

    def test_read_image(self, sample_image):
        client = LandmarkDiffClient()
        data = client._read_image(sample_image)
        assert len(data) > 0
        assert isinstance(data, bytes)
