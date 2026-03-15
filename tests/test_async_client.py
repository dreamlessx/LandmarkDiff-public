"""Tests for the async LandmarkDiff API client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from landmarkdiff.api_client import LandmarkDiffAPIError, PredictionResult
from landmarkdiff.api_client_async import AsyncLandmarkDiffClient


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAsyncClientInit:
    def test_default_url(self):
        client = AsyncLandmarkDiffClient()
        assert client.base_url == "http://localhost:8000"

    def test_custom_url_strips_trailing_slash(self):
        client = AsyncLandmarkDiffClient("http://example.com:9000/")
        assert client.base_url == "http://example.com:9000"

    def test_default_timeout(self):
        client = AsyncLandmarkDiffClient()
        assert client.timeout == 60.0

    def test_custom_timeout(self):
        client = AsyncLandmarkDiffClient(timeout=30.0)
        assert client.timeout == 30.0

    def test_max_concurrent(self):
        client = AsyncLandmarkDiffClient(max_concurrent=8)
        assert client.max_concurrent == 8

    def test_repr(self):
        client = AsyncLandmarkDiffClient("http://test:8000")
        assert "http://test:8000" in repr(client)


class TestAsyncClientImageIO:
    def test_read_image_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AsyncLandmarkDiffClient._read_image("/nonexistent/image.png")

    def test_read_image_from_file(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake_image_data")
        data = AsyncLandmarkDiffClient._read_image(img_path)
        assert data == b"fake_image_data"

    def test_decode_base64_invalid(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            AsyncLandmarkDiffClient._decode_base64_image("aW52YWxpZA==")


class TestAsyncClientSession:
    def test_requires_aiohttp(self):
        client = AsyncLandmarkDiffClient()
        with (
            patch.dict("sys.modules", {"aiohttp": None}),
            pytest.raises(ImportError, match="aiohttp"),
        ):
            _run(client._get_session())

    def test_close_noop_when_no_session(self):
        client = AsyncLandmarkDiffClient()
        _run(client.close())  # should not raise

    def test_context_manager(self):
        mock_session = AsyncMock()
        mock_session.closed = False

        async def _test():
            with patch.object(
                AsyncLandmarkDiffClient,
                "_get_session",
                return_value=mock_session,
            ):
                async with AsyncLandmarkDiffClient() as client:
                    client._session = mock_session
                mock_session.close.assert_called_once()

        _run(_test())


class TestAsyncClientHealth:
    def test_health_returns_dict(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"status": "ok", "version": "0.2.0"})

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=_AsyncContext(mock_resp))

        async def _test():
            client = AsyncLandmarkDiffClient()
            client._session = mock_session
            result = await client.health()
            assert result["status"] == "ok"

        _run(_test())

    def test_health_error(self):
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=_AsyncContext(mock_resp))

        async def _test():
            client = AsyncLandmarkDiffClient()
            client._session = mock_session
            with pytest.raises(LandmarkDiffAPIError, match="500"):
                await client.health()

        _run(_test())


class TestAsyncClientProcedures:
    def test_procedures_returns_list(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"procedures": ["rhinoplasty", "blepharoplasty"]})

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=_AsyncContext(mock_resp))

        async def _test():
            client = AsyncLandmarkDiffClient()
            client._session = mock_session
            result = await client.procedures()
            assert "rhinoplasty" in result
            assert len(result) == 2

        _run(_test())


class TestAsyncClientBatch:
    def test_batch_respects_max_concurrent(self):
        """Verify semaphore limits concurrent requests."""

        async def _test():
            client = AsyncLandmarkDiffClient(max_concurrent=2)
            active = 0
            max_active = 0

            async def mock_predict(path, **kwargs):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1
                return PredictionResult(
                    output_image=np.zeros((64, 64, 3), dtype=np.uint8),
                    procedure="rhinoplasty",
                    intensity=65.0,
                )

            client.predict = mock_predict
            paths = [f"/fake/img_{i}.png" for i in range(6)]
            results = await client.batch_predict(paths)
            assert len(results) == 6
            assert max_active <= 2

        _run(_test())

    def test_batch_preserves_order(self):
        """Results should match input order."""

        async def _test():
            client = AsyncLandmarkDiffClient(max_concurrent=2)

            async def mock_predict(path, **kwargs):
                idx = int(str(path).split("_")[1].split(".")[0])
                await asyncio.sleep(0.01 * (5 - idx))
                return PredictionResult(
                    output_image=np.full((64, 64, 3), idx, dtype=np.uint8),
                    procedure="rhinoplasty",
                    intensity=65.0,
                    metadata={"index": idx},
                )

            client.predict = mock_predict
            paths = [f"/fake/img_{i}.png" for i in range(5)]
            results = await client.batch_predict(paths)
            for i, r in enumerate(results):
                assert r.metadata["index"] == i

        _run(_test())

    def test_batch_handles_individual_failures(self):
        """Failed predictions should not block others."""

        async def _test():
            client = AsyncLandmarkDiffClient(max_concurrent=4)

            async def mock_predict(path, **kwargs):
                if "fail" in str(path):
                    raise LandmarkDiffAPIError("Server error")
                return PredictionResult(
                    output_image=np.zeros((64, 64, 3), dtype=np.uint8),
                    procedure="rhinoplasty",
                    intensity=65.0,
                )

            client.predict = mock_predict
            paths = ["/fake/ok.png", "/fake/fail.png", "/fake/also_ok.png"]
            results = await client.batch_predict(paths)
            assert len(results) == 3
            assert "error" not in results[0].metadata
            assert "error" in results[1].metadata
            assert "error" not in results[2].metadata

        _run(_test())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _AsyncContext:
    """Wrap an async mock to work as an async context manager."""

    def __init__(self, resp: AsyncMock) -> None:
        self.resp = resp

    async def __aenter__(self) -> AsyncMock:
        return self.resp

    async def __aexit__(self, *args: object) -> None:
        pass
