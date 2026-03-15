"""Async Python client for the LandmarkDiff REST API.

Provides an asyncio-compatible interface using aiohttp for non-blocking
interaction with the FastAPI server.

Usage:
    import asyncio
    from landmarkdiff.api_client_async import AsyncLandmarkDiffClient

    async def main():
        async with AsyncLandmarkDiffClient("http://localhost:8000") as client:
            health = await client.health()
            result = await client.predict("patient.png", procedure="rhinoplasty")
            result.save("output.png")

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from landmarkdiff.api_client import LandmarkDiffAPIError, PredictionResult

logger = logging.getLogger(__name__)


class AsyncLandmarkDiffClient:
    """Async client for the LandmarkDiff REST API.

    Uses aiohttp for non-blocking HTTP requests. Supports context
    manager usage and concurrent batch predictions.

    Args:
        base_url: Server URL (e.g. "http://localhost:8000").
        timeout: Request timeout in seconds.
        max_concurrent: Maximum concurrent requests for batch operations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_concurrent: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self._session: Any = None

    async def _get_session(self) -> Any:
        """Lazy-initialize aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
            except ImportError:
                raise ImportError(
                    "aiohttp required for async client. Install with: pip install aiohttp"
                ) from None
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @staticmethod
    def _read_image(image_path: str | Path) -> bytes:
        """Read image file as bytes."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()

    @staticmethod
    def _decode_base64_image(b64_string: str) -> np.ndarray:
        """Decode a base64-encoded image to numpy array."""
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image")
        return img

    async def _handle_response(self, resp: Any) -> dict[str, Any]:
        """Check response status and return JSON."""
        if resp.status >= 400:
            text = await resp.text()
            raise LandmarkDiffAPIError(f"Server returned {resp.status}: {text[:200]}")
        return await resp.json()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Dict with status and version info.
        """
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/health") as resp:
                return await self._handle_response(resp)
        except LandmarkDiffAPIError:
            raise
        except Exception as e:
            raise LandmarkDiffAPIError(
                f"Cannot connect to server at {self.base_url}: {e}"
            ) from None

    async def procedures(self) -> list[str]:
        """List available surgical procedures.

        Returns:
            List of procedure names.
        """
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/procedures") as resp:
                data = await self._handle_response(resp)
                return data.get("procedures", [])
        except LandmarkDiffAPIError:
            raise
        except Exception as e:
            raise LandmarkDiffAPIError(
                f"Cannot connect to server at {self.base_url}: {e}"
            ) from None

    async def predict(
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
        import aiohttp

        session = await self._get_session()
        image_bytes = self._read_image(image_path)

        data = aiohttp.FormData()
        data.add_field("image", image_bytes, filename="image.png", content_type="image/png")
        data.add_field("procedure", procedure)
        data.add_field("intensity", str(intensity))
        data.add_field("seed", str(seed))

        try:
            async with session.post(f"{self.base_url}/predict", data=data) as resp:
                result = await self._handle_response(resp)

                output_img = self._decode_base64_image(result["output_image"])

                return PredictionResult(
                    output_image=output_img,
                    procedure=procedure,
                    intensity=intensity,
                    confidence=result.get("confidence", 0.0),
                    metrics=result.get("metrics", {}),
                    metadata=result.get("metadata", {}),
                )
        except LandmarkDiffAPIError:
            raise
        except Exception as e:
            raise LandmarkDiffAPIError(f"Prediction failed: {e}") from None

    async def analyze(self, image_path: str | Path) -> dict[str, Any]:
        """Analyze a face image without generating a prediction.

        Args:
            image_path: Path to input face image.

        Returns:
            Dict with analysis results.
        """
        import aiohttp

        session = await self._get_session()
        image_bytes = self._read_image(image_path)

        data = aiohttp.FormData()
        data.add_field("image", image_bytes, filename="image.png", content_type="image/png")

        try:
            async with session.post(f"{self.base_url}/analyze", data=data) as resp:
                return await self._handle_response(resp)
        except LandmarkDiffAPIError:
            raise
        except Exception as e:
            raise LandmarkDiffAPIError(f"Analysis failed: {e}") from None

    async def batch_predict(
        self,
        image_paths: list[str | Path],
        procedure: str = "rhinoplasty",
        intensity: float = 65.0,
        seed: int = 42,
    ) -> list[PredictionResult]:
        """Run concurrent batch prediction on multiple images.

        Uses a semaphore to limit concurrent requests to max_concurrent.

        Args:
            image_paths: List of image file paths.
            procedure: Procedure to apply to all images.
            intensity: Intensity for all images.
            seed: Base random seed.

        Returns:
            List of PredictionResult objects in the same order as inputs.
        """
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _predict_one(path: str | Path, idx: int) -> PredictionResult:
            async with sem:
                try:
                    return await self.predict(
                        path,
                        procedure=procedure,
                        intensity=intensity,
                        seed=seed + idx,
                    )
                except Exception as e:
                    logger.warning("Batch prediction failed for %s: %s", path, e)
                    return PredictionResult(
                        output_image=np.zeros((512, 512, 3), dtype=np.uint8),
                        procedure=procedure,
                        intensity=intensity,
                        metadata={"error": str(e), "path": str(path)},
                    )

        tasks = [_predict_one(p, i) for i, p in enumerate(image_paths)]
        return list(await asyncio.gather(*tasks))

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> AsyncLandmarkDiffClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"AsyncLandmarkDiffClient(base_url='{self.base_url}')"
