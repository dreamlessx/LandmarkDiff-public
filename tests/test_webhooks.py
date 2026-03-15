"""Tests for webhook notification system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from landmarkdiff.webhooks import (
    EVENT_TYPES,
    WebhookNotifier,
    WebhookPayload,
)


class TestWebhookPayload:
    def test_creates_with_event_and_data(self):
        p = WebhookPayload(event="prediction.complete", data={"status": "ok"})
        assert p.event == "prediction.complete"
        assert p.data["status"] == "ok"

    def test_auto_generates_webhook_id(self):
        p = WebhookPayload(event="test", data={})
        assert len(p.webhook_id) == 12

    def test_unique_ids(self):
        ids = {WebhookPayload(event="test", data={}).webhook_id for _ in range(100)}
        assert len(ids) == 100

    def test_to_dict(self):
        p = WebhookPayload(event="test", data={"key": "val"})
        d = p.to_dict()
        assert d["event"] == "test"
        assert d["data"] == {"key": "val"}
        assert "webhook_id" in d
        assert "timestamp" in d

    def test_to_json_is_valid(self):
        import json

        p = WebhookPayload(event="test", data={"x": 1})
        parsed = json.loads(p.to_json())
        assert parsed["event"] == "test"


class TestEventTypes:
    def test_known_events(self):
        assert "prediction.complete" in EVENT_TYPES
        assert "batch.complete" in EVENT_TYPES
        assert "batch.failed" in EVENT_TYPES

    def test_event_count(self):
        assert len(EVENT_TYPES) == 9


class TestWebhookNotifierInit:
    def test_default_config(self):
        n = WebhookNotifier("https://example.com/hook")
        assert n.url == "https://example.com/hook"
        assert n.secret is None
        assert n.max_retries == 3
        assert n.timeout == 10.0

    def test_custom_config(self):
        n = WebhookNotifier(
            "https://example.com",
            secret="mysecret",
            max_retries=5,
            timeout=30.0,
            headers={"Authorization": "Bearer tok"},
        )
        assert n.secret == "mysecret"
        assert n.max_retries == 5
        assert n.headers["Authorization"] == "Bearer tok"


class TestWebhookSigning:
    def test_sign_produces_hex(self):
        n = WebhookNotifier("https://x.com", secret="test_secret")
        sig = n.sign('{"event":"test"}')
        assert len(sig) == 64  # SHA-256 hex digest

    def test_sign_requires_secret(self):
        n = WebhookNotifier("https://x.com")
        with pytest.raises(ValueError, match="No signing secret"):
            n.sign("body")

    def test_verify_correct_signature(self):
        n = WebhookNotifier("https://x.com", secret="s3cret")
        body = '{"event":"test"}'
        sig = n.sign(body)
        assert n.verify(body, sig) is True

    def test_verify_wrong_signature(self):
        n = WebhookNotifier("https://x.com", secret="s3cret")
        assert n.verify('{"event":"test"}', "wrong") is False

    def test_verify_without_secret(self):
        n = WebhookNotifier("https://x.com")
        assert n.verify("body", "sig") is False

    def test_different_bodies_different_sigs(self):
        n = WebhookNotifier("https://x.com", secret="key")
        sig1 = n.sign("body1")
        sig2 = n.sign("body2")
        assert sig1 != sig2


class TestWebhookSend:
    @patch("landmarkdiff.webhooks.time.sleep")
    def test_successful_delivery(self, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        n = WebhookNotifier("https://hook.example.com")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            delivery = n.send("prediction.complete", {"id": "abc"})

        assert delivery.success is True
        assert delivery.status_code == 200
        assert delivery.attempts == 1
        mock_post.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_retry_on_server_error(self, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200

        n = WebhookNotifier("https://hook.example.com", max_retries=3)
        with patch("requests.post", side_effect=[fail_resp, ok_resp]):
            delivery = n.send("test", {})

        assert delivery.success is True
        assert delivery.attempts == 2
        mock_sleep.assert_called_once_with(1)

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 503

        n = WebhookNotifier("https://hook.example.com", max_retries=3)
        with patch("requests.post", return_value=fail_resp):
            delivery = n.send("test", {})

        assert delivery.success is False
        assert delivery.attempts == 3
        assert mock_sleep.call_count == 2  # backoff between retries

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep):
        ok_resp = MagicMock()
        ok_resp.status_code = 200

        n = WebhookNotifier("https://hook.example.com", max_retries=3)
        with patch(
            "requests.post",
            side_effect=[ConnectionError("refused"), ok_resp],
        ):
            delivery = n.send("test", {})

        assert delivery.success is True
        assert delivery.attempts == 2

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_includes_signature_header(self, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        n = WebhookNotifier("https://hook.example.com", secret="key123")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            n.send("test", {"val": 1})

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert "X-Webhook-Signature" in headers
        assert "X-Webhook-Event" in headers

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_includes_custom_headers(self, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        n = WebhookNotifier(
            "https://hook.example.com",
            headers={"Authorization": "Bearer xyz"},
        )
        with patch("requests.post", return_value=mock_resp) as mock_post:
            n.send("test", {})

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert headers["Authorization"] == "Bearer xyz"

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_delivery_tracking(self, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        n = WebhookNotifier("https://hook.example.com")
        with patch("requests.post", return_value=mock_resp):
            n.send("event1", {})
            n.send("event2", {})

        assert len(n.deliveries) == 2
        assert n.deliveries[0].payload.event == "event1"
        assert n.deliveries[1].payload.event == "event2"


class TestWebhookMetrics:
    @patch("landmarkdiff.webhooks.time.sleep")
    def test_success_rate_all_ok(self, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        n = WebhookNotifier("https://hook.example.com")
        with patch("requests.post", return_value=mock_resp):
            n.send("e1", {})
            n.send("e2", {})

        assert n.success_rate == 1.0

    @patch("landmarkdiff.webhooks.time.sleep")
    def test_success_rate_mixed(self, mock_sleep):
        ok_resp = MagicMock()
        ok_resp.status_code = 200

        fail_resp = MagicMock()
        fail_resp.status_code = 500

        n = WebhookNotifier("https://hook.example.com", max_retries=1)
        with patch("requests.post", side_effect=[ok_resp, fail_resp]):
            n.send("ok", {})
            n.send("fail", {})

        assert n.success_rate == 0.5

    def test_success_rate_no_deliveries(self):
        n = WebhookNotifier("https://hook.example.com")
        assert n.success_rate == 0.0
