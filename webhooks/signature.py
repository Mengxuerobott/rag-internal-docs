"""
webhooks/signature.py
─────────────────────
HMAC-SHA256 signature verification for each supported DMS provider.

Every provider signs its webhook payload with a shared secret so the
receiver can verify the request genuinely came from that service and wasn't
forged by an attacker who learned your public webhook URL.

Verification pattern (same for all providers, different header names):
  1. Provider computes: HMAC-SHA256(secret, raw_request_body)
  2. Provider sends the hex digest in a request header.
  3. We re-compute the same HMAC over the raw body we received.
  4. We compare using hmac.compare_digest() (constant-time — prevents
     timing attacks where an attacker infers bytes by measuring response time).

If the secret is blank (empty string), verification is SKIPPED and a warning
is logged. This is intentional for local development / integration testing
where you haven't configured secrets yet — never run this way in production.

Provider-specific header names:
  Confluence  — X-Hub-Signature: sha256=<hex>
  SharePoint  — clientState header value is the secret itself (simpler model)
  Google Drive — X-Goog-Channel-Token: <token>
"""

import hashlib
import hmac

from fastapi import HTTPException, Request, status
from loguru import logger


async def _read_body_bytes(request: Request) -> bytes:
    """
    Read and cache the raw request body.

    FastAPI / Starlette consume the body stream on first read. We cache it
    in request.state so signature verification and JSON parsing can both
    access the same bytes without re-reading a consumed stream.
    """
    if not hasattr(request.state, "body"):
        request.state.body = await request.body()
    return request.state.body


def _compute_hmac(secret: str, payload: bytes) -> str:
    """Return the hex-encoded HMAC-SHA256 of payload using secret."""
    return hmac.new(
        secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()


def _constant_time_equal(a: str, b: str) -> bool:
    """Constant-time string comparison (prevents timing attacks)."""
    return hmac.compare_digest(a.encode(), b.encode())


# ── Confluence ────────────────────────────────────────────────────────────────
async def verify_confluence_signature(request: Request, secret: str) -> None:
    """
    Verify Confluence webhook HMAC.

    Confluence sends:  X-Hub-Signature: sha256=<hex_digest>
    Docs: https://developer.atlassian.com/server/confluence/webhooks/
    """
    if not secret:
        logger.warning("WEBHOOK_SECRET_CONFLUENCE not set — skipping verification (dev mode)")
        return

    header = request.headers.get("X-Hub-Signature", "")
    if not header.startswith("sha256="):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed X-Hub-Signature header",
        )

    received_hex = header[len("sha256="):]
    body = await _read_body_bytes(request)
    expected_hex = _compute_hmac(secret, body)

    if not _constant_time_equal(received_hex, expected_hex):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Confluence webhook signature mismatch",
        )


# ── SharePoint ────────────────────────────────────────────────────────────────
async def verify_sharepoint_signature(request: Request, secret: str) -> None:
    """
    Verify SharePoint webhook authenticity.

    SharePoint's change-notification model sends a `clientState` field in the
    JSON body that matches the secret you set when creating the subscription.
    This is a simpler token-equality check rather than an HMAC, but we also
    support an X-Hub-Signature header for custom middleware setups.

    Docs: https://learn.microsoft.com/en-us/sharepoint/dev/apis/webhooks/overview-sharepoint-webhooks
    """
    if not secret:
        logger.warning("WEBHOOK_SECRET_SHAREPOINT not set — skipping verification (dev mode)")
        return

    body = await _read_body_bytes(request)

    # Try X-Hub-Signature first (custom reverse-proxy setup)
    hub_sig = request.headers.get("X-Hub-Signature", "")
    if hub_sig.startswith("sha256="):
        received_hex = hub_sig[len("sha256="):]
        expected_hex = _compute_hmac(secret, body)
        if not _constant_time_equal(received_hex, expected_hex):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="SharePoint webhook signature mismatch",
            )
        return

    # Fall back to clientState field in JSON body
    import json
    try:
        payload = json.loads(body)
        # SharePoint wraps notifications in a "value" array
        notifications = payload.get("value", [payload])
        for notif in notifications:
            client_state = notif.get("clientState", "")
            if not _constant_time_equal(client_state, secret):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="SharePoint clientState mismatch",
                )
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid SharePoint webhook payload",
        )


# ── Google Drive ──────────────────────────────────────────────────────────────
async def verify_gdrive_signature(request: Request, secret: str) -> None:
    """
    Verify Google Drive push-notification channel token.

    Google sends:  X-Goog-Channel-Token: <token>
    The token is the value you set in the `token` field when creating the
    push-notification channel via the Drive API.

    This is a simple constant-time equality check (no HMAC because Google
    doesn't sign the body — the token acts as a bearer credential).

    Docs: https://developers.google.com/drive/api/guides/push#understanding-push-notifications
    """
    if not secret:
        logger.warning("WEBHOOK_SECRET_GDRIVE not set — skipping verification (dev mode)")
        return

    received_token = request.headers.get("X-Goog-Channel-Token", "")
    if not received_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Goog-Channel-Token header",
        )

    if not _constant_time_equal(received_token, secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google Drive channel token mismatch",
        )
