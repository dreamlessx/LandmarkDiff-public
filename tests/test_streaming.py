"""Tests for WebSocket streaming protocol."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from landmarkdiff.streaming import (
    FrameMessage,
    MessageType,
    StreamConfig,
    StreamSession,
    encode_frame,
)


@pytest.fixture
def sample_images():
    """Create a pair of small test images."""
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    original[:, :, 2] = 200  # red
    prediction = np.zeros((64, 64, 3), dtype=np.uint8)
    prediction[:, :, 0] = 200  # blue
    return original, prediction


class TestMessageType:
    def test_all_types_are_strings(self):
        for mt in MessageType:
            assert isinstance(mt.value, str)

    def test_expected_types_exist(self):
        assert MessageType.FRAME.value == "frame"
        assert MessageType.COMPLETE.value == "complete"
        assert MessageType.ERROR.value == "error"
        assert MessageType.PROGRESS.value == "progress"


class TestStreamConfig:
    def test_defaults(self):
        cfg = StreamConfig()
        assert cfg.procedure == "rhinoplasty"
        assert cfg.intensity == 65.0
        assert cfg.n_preview_frames == 5
        assert cfg.quality == 80

    def test_from_dict(self):
        cfg = StreamConfig.from_dict(
            {
                "procedure": "blepharoplasty",
                "intensity": 80.0,
                "quality": 90,
            }
        )
        assert cfg.procedure == "blepharoplasty"
        assert cfg.intensity == 80.0
        assert cfg.quality == 90

    def test_from_dict_ignores_unknown(self):
        cfg = StreamConfig.from_dict(
            {
                "procedure": "rhinoplasty",
                "unknown_field": "ignored",
            }
        )
        assert cfg.procedure == "rhinoplasty"
        assert not hasattr(cfg, "unknown_field")


class TestFrameMessage:
    def test_frame_to_dict(self):
        msg = FrameMessage(
            msg_type=MessageType.FRAME,
            session_id="abc123",
            frame_index=2,
            total_frames=5,
            progress=0.4,
            image_b64="base64data",
        )
        d = msg.to_dict()
        assert d["type"] == "frame"
        assert d["session_id"] == "abc123"
        assert d["frame_index"] == 2
        assert d["image"] == "base64data"
        assert d["progress"] == 0.4

    def test_progress_to_dict(self):
        msg = FrameMessage(
            msg_type=MessageType.PROGRESS,
            session_id="abc123",
            progress=0.75,
            stage="diffusion_inference",
            description="Running denoising",
        )
        d = msg.to_dict()
        assert d["type"] == "progress"
        assert d["stage"] == "diffusion_inference"
        assert d["progress"] == 0.75

    def test_error_to_dict(self):
        msg = FrameMessage(
            msg_type=MessageType.ERROR,
            session_id="abc123",
            error="No face detected",
        )
        d = msg.to_dict()
        assert d["type"] == "error"
        assert d["error"] == "No face detected"

    def test_complete_to_dict(self):
        msg = FrameMessage(
            msg_type=MessageType.COMPLETE,
            session_id="abc123",
            image_b64="final_image",
            metadata={"procedure": "rhinoplasty"},
        )
        d = msg.to_dict()
        assert d["type"] == "complete"
        assert d["image"] == "final_image"
        assert d["metadata"]["procedure"] == "rhinoplasty"

    def test_to_dict_is_json_serializable(self):
        msg = FrameMessage(
            msg_type=MessageType.FRAME,
            session_id="test",
            frame_index=0,
            total_frames=3,
            image_b64="data",
        )
        serialized = json.dumps(msg.to_dict())
        assert isinstance(serialized, str)


class TestEncodeFrame:
    def test_returns_base64_string(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        b64 = encode_frame(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_decodable(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        b64 = encode_frame(img)
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_quality_affects_size(self):
        img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        low = encode_frame(img, quality=10)
        high = encode_frame(img, quality=95)
        assert len(low) < len(high)

    def test_clamps_quality(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        # should not raise with out-of-range quality
        encode_frame(img, quality=0)
        encode_frame(img, quality=200)


class TestStreamSession:
    def test_creates_with_defaults(self):
        session = StreamSession()
        assert session.config.procedure == "rhinoplasty"
        assert len(session.session_id) == 12

    def test_creates_from_dict(self):
        session = StreamSession.from_dict(
            {
                "config": {"procedure": "blepharoplasty", "intensity": 40.0},
                "session_id": "custom_id",
            }
        )
        assert session.config.procedure == "blepharoplasty"
        assert session.session_id == "custom_id"

    def test_generate_preview_frames_count(self, sample_images):
        original, prediction = sample_images
        config = StreamConfig(n_preview_frames=3)
        session = StreamSession(config=config)
        messages = session.generate_preview_frames(original, prediction)
        # 3 frames + 1 complete
        assert len(messages) == 4
        assert messages[-1].msg_type == MessageType.COMPLETE

    def test_frame_progress_increases(self, sample_images):
        original, prediction = sample_images
        config = StreamConfig(n_preview_frames=5)
        session = StreamSession(config=config)
        messages = session.generate_preview_frames(original, prediction)
        frame_msgs = [m for m in messages if m.msg_type == MessageType.FRAME]
        progresses = [m.progress for m in frame_msgs]
        assert progresses == sorted(progresses)
        assert progresses[-1] == pytest.approx(1.0)

    def test_frames_have_images(self, sample_images):
        original, prediction = sample_images
        config = StreamConfig(n_preview_frames=2)
        session = StreamSession(config=config)
        messages = session.generate_preview_frames(original, prediction)
        for msg in messages:
            if msg.msg_type in (MessageType.FRAME, MessageType.COMPLETE):
                assert len(msg.image_b64) > 0

    def test_cancel_stops_generation(self, sample_images):
        original, prediction = sample_images
        config = StreamConfig(n_preview_frames=10)
        session = StreamSession(config=config)
        session.cancel()
        messages = session.generate_preview_frames(original, prediction)
        assert len(messages) == 0  # cancelled before any frames

    def test_shape_mismatch_returns_error(self):
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((32, 32, 3), dtype=np.uint8)
        session = StreamSession()
        messages = session.generate_preview_frames(img1, img2)
        assert len(messages) == 1
        assert messages[0].msg_type == MessageType.ERROR
        assert "mismatch" in messages[0].error.lower()

    def test_complete_message_has_metadata(self, sample_images):
        original, prediction = sample_images
        config = StreamConfig(
            procedure="orthognathic",
            intensity=70.0,
            n_preview_frames=1,
        )
        session = StreamSession(config=config)
        messages = session.generate_preview_frames(original, prediction)
        complete = messages[-1]
        assert complete.metadata["procedure"] == "orthognathic"
        assert complete.metadata["intensity"] == 70.0

    def test_make_progress_message(self):
        session = StreamSession()
        msg = session.make_progress_message(0.5, "inference", "Running diffusion")
        assert msg.msg_type == MessageType.PROGRESS
        assert msg.progress == 0.5
        assert msg.stage == "inference"
        assert msg.session_id == session.session_id

    def test_make_error_message(self):
        session = StreamSession()
        msg = session.make_error_message("GPU out of memory")
        assert msg.msg_type == MessageType.ERROR
        assert msg.error == "GPU out of memory"
