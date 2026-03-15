"""Webhook notification system for batch and pipeline events.

Sends HTTP POST callbacks when predictions complete, fail, or when
batch jobs finish. Supports retry with exponential backoff and
optional HMAC-SHA256 signature verification.

Usage:
    from landmarkdiff.webhooks import WebhookNotifier

    notifier = WebhookNotifier("https://example.com/webhook")
    notifier.send("prediction.complete", {
        "image_id": "patient_001",
        "procedure": "rhinoplasty",
        "status": "success",
    })
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Recognized event types
EVENT_TYPES = frozenset(
    {
        "prediction.started",
        "prediction.complete",
        "prediction.failed",
        "batch.started",
        "batch.progress",
        "batch.complete",
        "batch.failed",
        "analysis.complete",
        "health.degraded",
    }
)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TIMEOUT = 10.0


@dataclass
class WebhookPayload:
    """Structured webhook payload."""

    event: str
    data: dict[str, Any]
    webhook_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "data": self.data,
            "webhook_id": self.webhook_id,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    payload: WebhookPayload
    status_code: int = 0
    success: bool = False
    attempts: int = 0
    error: str = ""
    duration_ms: float = 0.0


class WebhookNotifier:
    """Send webhook notifications with retry and optional signing.

    Args:
        url: Webhook endpoint URL.
        secret: Optional HMAC-SHA256 signing secret.
        max_retries: Maximum delivery attempts (default 3).
        timeout: HTTP timeout in seconds (default 10).
        headers: Extra HTTP headers to include.
    """

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.secret = secret
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = headers or {}
        self._deliveries: list[WebhookDelivery] = []

    def sign(self, body: str) -> str:
        """Compute HMAC-SHA256 signature for a payload body.

        Args:
            body: JSON string to sign.

        Returns:
            Hex-encoded HMAC-SHA256 signature.

        Raises:
            ValueError: If no secret is configured.
        """
        if not self.secret:
            raise ValueError("No signing secret configured")
        return hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()

    def verify(self, body: str, signature: str) -> bool:
        """Verify an HMAC-SHA256 signature.

        Args:
            body: JSON string that was signed.
            signature: Hex-encoded signature to verify.

        Returns:
            True if signature matches.
        """
        if not self.secret:
            return False
        expected = self.sign(body)
        return hmac.compare_digest(expected, signature)

    def send(
        self,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> WebhookDelivery:
        """Send a webhook notification.

        Retries with exponential backoff on failure (1s, 2s, 4s, ...).

        Args:
            event: Event type (e.g. "prediction.complete").
            data: Event-specific data payload.

        Returns:
            WebhookDelivery with status and attempt details.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests required for webhooks. Install with: pip install requests"
            ) from None

        payload = WebhookPayload(event=event, data=data or {})
        body = payload.to_json()

        request_headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event,
            "X-Webhook-ID": payload.webhook_id,
            **self.headers,
        }
        if self.secret:
            request_headers["X-Webhook-Signature"] = self.sign(body)

        delivery = WebhookDelivery(payload=payload)

        for attempt in range(self.max_retries):
            delivery.attempts = attempt + 1
            start = time.monotonic()
            try:
                resp = requests.post(
                    self.url,
                    data=body,
                    headers=request_headers,
                    timeout=self.timeout,
                )
                delivery.status_code = resp.status_code
                delivery.duration_ms = (time.monotonic() - start) * 1000
                delivery.success = 200 <= resp.status_code < 300
                if delivery.success:
                    logger.info(
                        "Webhook %s delivered (%d ms)",
                        payload.webhook_id,
                        int(delivery.duration_ms),
                    )
                    break
                delivery.error = f"HTTP {resp.status_code}"
            except Exception as e:
                delivery.duration_ms = (time.monotonic() - start) * 1000
                delivery.error = str(e)

            if attempt < self.max_retries - 1:
                backoff = 2**attempt
                logger.warning(
                    "Webhook %s attempt %d failed (%s), retrying in %ds",
                    payload.webhook_id,
                    attempt + 1,
                    delivery.error,
                    backoff,
                )
                time.sleep(backoff)

        if not delivery.success:
            logger.error(
                "Webhook %s failed after %d attempts: %s",
                payload.webhook_id,
                delivery.attempts,
                delivery.error,
            )

        self._deliveries.append(delivery)
        return delivery

    @property
    def deliveries(self) -> list[WebhookDelivery]:
        """All delivery attempts in chronological order."""
        return list(self._deliveries)

    @property
    def success_rate(self) -> float:
        """Fraction of successful deliveries (0.0 - 1.0)."""
        if not self._deliveries:
            return 0.0
        successes = sum(1 for d in self._deliveries if d.success)
        return successes / len(self._deliveries)
