"""Python client for the LandmarkDiff REST API.

Provides a clean interface for interacting with the FastAPI server,
handling image encoding/decoding, error handling, and session management.

Usage:
    from landmarkdiff.api_client import LandmarkDiffClient

    client = LandmarkDiffClient("http://localhost:8000")

    # Single prediction
    result = client.predict("patient.png", procedure="rhinoplasty", intensity=65)
    result.save("output.png")

    # Face analysis
    analysis = client.analyze("patient.png")
    print(f"Fitzpatrick type: {analysis['fitzpatrick_type']}")

    # Batch processing
    results = client.batch_predict(
        ["patient1.png", "patient2.png"],
        procedure="blepharoplasty",
    )
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    output_image: np.ndarray
    procedure: str
    intensity: float
    confidence: float = 0.0
    landmarks_before: list | None = None
    landmarks_after: list | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path, fmt: str = ".png") -> None:
        """Save the output image to a file."""
        cv2.imwrite(str(path), self.output_image)

    def show(self) -> None:
        """Display the output image (requires GUI)."""
        cv2.imshow("LandmarkDiff Prediction", self.output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class LandmarkDiffClient:
    """Client for the LandmarkDiff REST API.

    Args:
        base_url: Server URL (e.g. "http://localhost:8000").
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        """Lazy-initialize requests session."""
        if self._session is None:
            try:
                import requests
            except ImportError:
                raise ImportError("requests required. Install with: pip install requests")
            self._session = requests.Session()
            self._session.timeout = self.timeout
        return self._session

    def _read_image(self, image_path: str | Path) -> bytes:
        """Read image file as bytes."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()

    def _decode_base64_image(self, b64_string: str) -> np.ndarray:
        """Decode a base64-encoded image to numpy array."""
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image")
        return img

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Dict with status and version info.
        """
        session = self._get_session()
        resp = session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def procedures(self) -> list[str]:
        """List available surgical procedures.

        Returns:
            List of procedure names.
        """
        session = self._get_session()
        resp = session.get(f"{self.base_url}/procedures")
        resp.raise_for_status()
        return resp.json().get("procedures", [])

    def predict(
        self,
        image_path: str | Path,
        procedure: str = "rhinoplasty",
        intensity: float = 65.0,
        seed: int = 42,
    ) -> PredictionResult:
        """Run surgical outcome prediction.

        Args:
            image_path: Path to input face image.
            procedure: Surgical procedure type.
            intensity: Intensity of the modification (0-100).
            seed: Random seed for reproducibility.

        Returns:
            PredictionResult with output image and metadata.
        """
        session = self._get_session()
        image_bytes = self._read_image(image_path)

        files = {"image": ("image.png", image_bytes, "image/png")}
        data = {
            "procedure": procedure,
            "intensity": str(intensity),
            "seed": str(seed),
        }

        resp = session.post(f"{self.base_url}/predict", files=files, data=data)
        resp.raise_for_status()
        result = resp.json()

        # Decode output image
        output_img = self._decode_base64_image(result["output_image"])

        return PredictionResult(
            output_image=output_img,
            procedure=procedure,
            intensity=intensity,
            confidence=result.get("confidence", 0.0),
            metrics=result.get("metrics", {}),
            metadata=result.get("metadata", {}),
        )

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        """Analyze a face image without generating a prediction.

        Returns face landmarks, Fitzpatrick type, pose estimation, etc.

        Args:
            image_path: Path to input face image.

        Returns:
            Dict with analysis results.
        """
        session = self._get_session()
        image_bytes = self._read_image(image_path)

        files = {"image": ("image.png", image_bytes, "image/png")}
        resp = session.post(f"{self.base_url}/analyze", files=files)
        resp.raise_for_status()
        return resp.json()

    def batch_predict(
        self,
        image_paths: list[str | Path],
        procedure: str = "rhinoplasty",
        intensity: float = 65.0,
        seed: int = 42,
    ) -> list[PredictionResult]:
        """Run batch prediction on multiple images.

        Args:
            image_paths: List of image file paths.
            procedure: Procedure to apply to all images.
            intensity: Intensity for all images.
            seed: Base random seed.

        Returns:
            List of PredictionResult objects.
        """
        results = []
        for i, path in enumerate(image_paths):
            try:
                result = self.predict(
                    path,
                    procedure=procedure,
                    intensity=intensity,
                    seed=seed + i,
                )
                results.append(result)
            except Exception as e:
                # Create a failed result
                results.append(PredictionResult(
                    output_image=np.zeros((512, 512, 3), dtype=np.uint8),
                    procedure=procedure,
                    intensity=intensity,
                    metadata={"error": str(e), "path": str(path)},
                ))
        return results

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        return f"LandmarkDiffClient(base_url='{self.base_url}')"
