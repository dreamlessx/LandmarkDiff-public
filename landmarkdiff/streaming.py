"""WebSocket streaming protocol for real-time prediction preview.

Defines the message protocol and frame generator for streaming
progressive prediction results over WebSocket connections.

Usage with FastAPI:
    from fastapi import WebSocket
    from landmarkdiff.streaming import StreamSession, FrameMessage

    @app.websocket("/ws/predict")
    async def ws_predict(websocket: WebSocket):
        await websocket.accept()
        session = StreamSession.from_websocket_params(await websocket.receive_json())
        async for frame_msg in session.generate_frames():
            await websocket.send_json(frame_msg.to_dict())
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types for the streaming protocol."""

    CONNECT = "connect"
    START = "start"
    FRAME = "frame"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ERROR = "error"
    CANCEL = "cancel"


@dataclass
class StreamConfig:
    """Configuration for a streaming prediction session.

    Args:
        procedure: Surgical procedure name.
        intensity: Target intensity (0-100).
        n_preview_frames: Number of intermediate preview frames.
        preview_interval_ms: Minimum time between preview frames.
        resolution: Image resolution.
        seed: Random seed.
        quality: JPEG quality for frame encoding (1-100).
    """

    procedure: str = "rhinoplasty"
    intensity: float = 65.0
    n_preview_frames: int = 5
    preview_interval_ms: int = 200
    resolution: int = 512
    seed: int = 42
    quality: int = 80

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamConfig:
        """Create config from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class FrameMessage:
    """A single frame message in the streaming protocol."""

    msg_type: MessageType
    session_id: str
    frame_index: int = 0
    total_frames: int = 0
    progress: float = 0.0
    image_b64: str = ""
    stage: str = ""
    description: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.msg_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }
        if self.msg_type == MessageType.FRAME:
            d["frame_index"] = self.frame_index
            d["total_frames"] = self.total_frames
            d["image"] = self.image_b64
            d["progress"] = round(self.progress, 3)
        elif self.msg_type == MessageType.PROGRESS:
            d["progress"] = round(self.progress, 3)
            d["stage"] = self.stage
            d["description"] = self.description
        elif self.msg_type == MessageType.ERROR:
            d["error"] = self.error
        elif self.msg_type == MessageType.COMPLETE:
            d["image"] = self.image_b64
            d["metadata"] = self.metadata
        return d


def encode_frame(image: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR image as a base64 JPEG string.

    Args:
        image: BGR image array.
        quality: JPEG quality (1-100).

    Returns:
        Base64-encoded JPEG string.
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, quality))]
    _, buf = cv2.imencode(".jpg", image, params)
    return base64.b64encode(buf.tobytes()).decode("ascii")


class StreamSession:
    """Manages a single streaming prediction session.

    Generates progressive preview frames by interpolating between
    the original and predicted images, sending them as WebSocket
    messages at the configured interval.

    Args:
        config: Stream configuration.
        session_id: Unique session identifier (auto-generated if not given).
    """

    def __init__(
        self,
        config: StreamConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config or StreamConfig()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.cancelled = False
        self._start_time = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamSession:
        """Create a session from WebSocket message data."""
        config = StreamConfig.from_dict(data.get("config", data))
        return cls(config=config, session_id=data.get("session_id"))

    def cancel(self) -> None:
        """Cancel the streaming session."""
        self.cancelled = True

    def generate_preview_frames(
        self,
        original: np.ndarray,
        prediction: np.ndarray,
    ) -> list[FrameMessage]:
        """Generate a sequence of progressive preview frame messages.

        Creates alpha-blended intermediate frames between original and
        prediction, encoding each as a JPEG for WebSocket transmission.

        Args:
            original: BGR original image.
            prediction: BGR predicted image.

        Returns:
            List of FrameMessage objects for streaming.
        """
        if original.shape != prediction.shape:
            return [
                FrameMessage(
                    msg_type=MessageType.ERROR,
                    session_id=self.session_id,
                    error=f"Shape mismatch: {original.shape} vs {prediction.shape}",
                )
            ]

        n = self.config.n_preview_frames
        if n < 1:
            n = 1

        orig_f = original.astype(np.float32)
        pred_f = prediction.astype(np.float32)
        messages: list[FrameMessage] = []

        for i in range(n):
            if self.cancelled:
                break

            alpha = (i + 1) / n
            blended = np.clip(orig_f * (1.0 - alpha) + pred_f * alpha, 0, 255).astype(np.uint8)

            messages.append(
                FrameMessage(
                    msg_type=MessageType.FRAME,
                    session_id=self.session_id,
                    frame_index=i,
                    total_frames=n,
                    progress=alpha,
                    image_b64=encode_frame(blended, self.config.quality),
                )
            )

        if not self.cancelled:
            messages.append(
                FrameMessage(
                    msg_type=MessageType.COMPLETE,
                    session_id=self.session_id,
                    image_b64=encode_frame(prediction, self.config.quality),
                    metadata={
                        "procedure": self.config.procedure,
                        "intensity": self.config.intensity,
                        "resolution": self.config.resolution,
                    },
                )
            )

        return messages

    def make_progress_message(
        self,
        progress: float,
        stage: str,
        description: str = "",
    ) -> FrameMessage:
        """Create a progress update message.

        Args:
            progress: Completion fraction (0.0 - 1.0).
            stage: Current pipeline stage name.
            description: Human-readable description.

        Returns:
            A progress-type FrameMessage.
        """
        return FrameMessage(
            msg_type=MessageType.PROGRESS,
            session_id=self.session_id,
            progress=progress,
            stage=stage,
            description=description,
        )

    def make_error_message(self, error: str) -> FrameMessage:
        """Create an error message.

        Args:
            error: Error description.

        Returns:
            An error-type FrameMessage.
        """
        return FrameMessage(
            msg_type=MessageType.ERROR,
            session_id=self.session_id,
            error=error,
        )
