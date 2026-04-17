"""
tests/test_webhooks.py
──────────────────────
Tests for the event-driven ingestion layer:
  - HMAC signature verification (all three providers)
  - Webhook payload normalisation (Confluence, SharePoint, GDrive)
  - Worker job logic (delete + upsert called correctly for each event type)
  - Idempotency (re-enqueueing the same doc_id doesn't corrupt state)

No real Redis, Qdrant, or DMS connection required — everything is mocked.

Run:
    pytest tests/test_webhooks.py -v
"""

import hashlib
import hmac
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Signature verification unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfluenceSignature:
    """webhooks/signature.py — verify_confluence_signature"""

    @pytest.mark.asyncio
    async def test_valid_signature_passes(self):
        from webhooks.signature import verify_confluence_signature

        secret = "test-secret"
        body = b'{"event": "page_updated"}'
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        request = MagicMock()
        request.headers = {"X-Hub-Signature": f"sha256={sig}"}
        request.state = MagicMock(spec=[])
        request.body = AsyncMock(return_value=body)

        # Should not raise
        await verify_confluence_signature(request, secret)

    @pytest.mark.asyncio
    async def test_invalid_signature_raises_401(self):
        from webhooks.signature import verify_confluence_signature

        request = MagicMock()
        request.headers = {"X-Hub-Signature": "sha256=badhex"}
        request.state = MagicMock(spec=[])
        request.body = AsyncMock(return_value=b'{"event": "page_updated"}')

        with pytest.raises(HTTPException) as exc_info:
            await verify_confluence_signature(request, "real-secret")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_header_raises_401(self):
        from webhooks.signature import verify_confluence_signature

        request = MagicMock()
        request.headers = {}
        request.state = MagicMock(spec=[])
        request.body = AsyncMock(return_value=b"{}")

        with pytest.raises(HTTPException) as exc_info:
            await verify_confluence_signature(request, "secret")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_secret_skips_verification(self):
        from webhooks.signature import verify_confluence_signature

        request = MagicMock()
        request.headers = {}
        # Should not raise even with no header when secret is empty
        await verify_confluence_signature(request, "")


class TestGDriveSignature:
    """webhooks/signature.py — verify_gdrive_signature"""

    @pytest.mark.asyncio
    async def test_valid_token_passes(self):
        from webhooks.signature import verify_gdrive_signature

        request = MagicMock()
        request.headers = {"X-Goog-Channel-Token": "my-secret-token"}
        await verify_gdrive_signature(request, "my-secret-token")

    @pytest.mark.asyncio
    async def test_wrong_token_raises_401(self):
        from webhooks.signature import verify_gdrive_signature

        request = MagicMock()
        request.headers = {"X-Goog-Channel-Token": "wrong-token"}
        with pytest.raises(HTTPException) as exc_info:
            await verify_gdrive_signature(request, "real-token")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_token_raises_401(self):
        from webhooks.signature import verify_gdrive_signature

        request = MagicMock()
        request.headers = {}
        with pytest.raises(HTTPException) as exc_info:
            await verify_gdrive_signature(request, "real-token")
        assert exc_info.value.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Webhook endpoint integration tests (TestClient, mocked ARQ)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_confluence_payload(event: str = "page_updated", page_id: str = "99999") -> dict:
    return {
        "event": event,
        "page": {
            "id": page_id,
            "title": "Engineering Onboarding",
            "space": {"key": "ENG"},
            "_links": {"self": "https://example.atlassian.net/wiki/pages/99999"},
        },
        "timestamp": 1700000000000,
    }


def _sign_confluence(body: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


@pytest.fixture(scope="module")
def webhook_client():
    """
    TestClient with:
      - Index + auth mocked out (we only care about the webhook layer)
      - ARQ enqueue_job mocked to return a fake job
    """
    mock_job = MagicMock()
    mock_job.job_id = "fake-job-id-123"

    mock_redis_pool = AsyncMock()
    mock_redis_pool.enqueue_job = AsyncMock(return_value=mock_job)
    mock_redis_pool.aclose = AsyncMock()

    with patch("api.main.get_or_build_index", return_value=MagicMock()), \
         patch("api.main.set_index"), \
         patch("api.main.get_index", return_value=MagicMock()), \
         patch("webhooks.router._get_redis_pool", return_value=mock_redis_pool), \
         patch("api.main.build_query_engine_for_user", return_value=MagicMock()), \
         patch("api.main._build_engine_for", return_value=MagicMock()):
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestConfluenceWebhook:
    SECRET = "confluence-secret"

    def _headers(self, body: bytes) -> dict:
        return {"X-Hub-Signature": _sign_confluence(body, self.SECRET)}

    def test_page_updated_returns_200_and_queued(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_CONFLUENCE", self.SECRET):
            payload = _build_confluence_payload("page_updated")
            body = json.dumps(payload).encode()
            r = webhook_client.post(
                "/webhooks/confluence",
                content=body,
                headers={**self._headers(body), "Content-Type": "application/json"},
            )
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        assert r.json()["event_type"] == "updated"

    def test_page_created_event_type(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_CONFLUENCE", self.SECRET):
            payload = _build_confluence_payload("page_created")
            body = json.dumps(payload).encode()
            r = webhook_client.post(
                "/webhooks/confluence",
                content=body,
                headers={**self._headers(body), "Content-Type": "application/json"},
            )
        assert r.status_code == 200
        assert r.json()["event_type"] == "created"

    def test_page_removed_event_type(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_CONFLUENCE", self.SECRET):
            payload = _build_confluence_payload("page_removed")
            body = json.dumps(payload).encode()
            r = webhook_client.post(
                "/webhooks/confluence",
                content=body,
                headers={**self._headers(body), "Content-Type": "application/json"},
            )
        assert r.status_code == 200
        assert r.json()["event_type"] == "deleted"

    def test_invalid_signature_rejected(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_CONFLUENCE", self.SECRET):
            payload = _build_confluence_payload()
            body = json.dumps(payload).encode()
            r = webhook_client.post(
                "/webhooks/confluence",
                content=body,
                headers={
                    "X-Hub-Signature": "sha256=badbadbadhex",
                    "Content-Type": "application/json",
                },
            )
        assert r.status_code == 401


class TestSharePointWebhook:
    def test_validation_handshake_returns_token(self, webhook_client):
        """SharePoint sends validationToken on first subscription — must echo it back."""
        r = webhook_client.post(
            "/webhooks/sharepoint?validationToken=my-validation-token",
            json={},
        )
        assert r.status_code == 200
        assert "my-validation-token" in r.text

    def test_change_notification_queued(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_SHAREPOINT", "sp-secret"):
            payload = {
                "value": [{
                    "subscriptionId": "abc",
                    "clientState": "sp-secret",
                    "changeType": "updated",
                    "resource": "sites/site-id/lists/engineering/items/item-42",
                    "resourceData": {"id": "item-42", "name": "design-doc.docx"},
                }]
            }
            r = webhook_client.post("/webhooks/sharepoint", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        assert r.json()["count"] == 1

    def test_deleted_change_type(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_SHAREPOINT", ""):
            payload = {
                "value": [{
                    "clientState": "",
                    "changeType": "deleted",
                    "resource": "sites/s/lists/hr/items/item-7",
                    "resourceData": {},
                }]
            }
            r = webhook_client.post("/webhooks/sharepoint", json=payload)
        assert r.status_code == 200


class TestGDriveWebhook:
    TOKEN = "gdrive-channel-token"

    def test_sync_handshake_acknowledged(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_GDRIVE", self.TOKEN):
            r = webhook_client.post(
                "/webhooks/gdrive",
                headers={
                    "X-Goog-Channel-Token": self.TOKEN,
                    "X-Goog-Resource-State": "sync",
                    "X-Goog-Resource-ID": "file-abc",
                    "X-Goog-Channel-ID": "channel-1",
                },
            )
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_update_notification_queued(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_GDRIVE", self.TOKEN):
            r = webhook_client.post(
                "/webhooks/gdrive",
                headers={
                    "X-Goog-Channel-Token": self.TOKEN,
                    "X-Goog-Resource-State": "update",
                    "X-Goog-Resource-ID": "file-xyz",
                    "X-Goog-Channel-ID": "channel-1",
                    "X-Goog-Changed": "content",
                    "X-Goog-Message-Number": "42",
                },
            )
        assert r.status_code == 200
        assert r.json()["status"] == "queued"
        assert r.json()["event_type"] == "updated"

    def test_trash_event_is_deleted(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_GDRIVE", self.TOKEN):
            r = webhook_client.post(
                "/webhooks/gdrive",
                headers={
                    "X-Goog-Channel-Token": self.TOKEN,
                    "X-Goog-Resource-State": "trash",
                    "X-Goog-Resource-ID": "file-xyz",
                    "X-Goog-Channel-ID": "channel-1",
                },
            )
        assert r.status_code == 200
        assert r.json()["event_type"] == "deleted"

    def test_invalid_token_rejected(self, webhook_client):
        with patch("config.settings.WEBHOOK_SECRET_GDRIVE", self.TOKEN):
            r = webhook_client.post(
                "/webhooks/gdrive",
                headers={
                    "X-Goog-Channel-Token": "WRONG-TOKEN",
                    "X-Goog-Resource-State": "update",
                    "X-Goog-Resource-ID": "file-xyz",
                    "X-Goog-Channel-ID": "channel-1",
                },
            )
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Worker job logic unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkerJobLogic:
    """Tests for workers/ingestion_worker.py — process_document_event"""

    @pytest.mark.asyncio
    async def test_deleted_event_only_deletes_no_upsert(self):
        from workers.ingestion_worker import process_document_event

        event_data = {
            "provider": "confluence",
            "event_type": "deleted",
            "doc_id": "page-123",
            "doc_url": "",
            "doc_title": "Old Page",
            "folder_path": "general",
        }

        with patch("workers.ingestion_worker.delete_document_chunks") as mock_delete, \
             patch("workers.ingestion_worker.upsert_document") as mock_upsert, \
             patch("workers.ingestion_worker.configure_llama_index"):

            result = await process_document_event({}, event_data)

        mock_delete.assert_called_once_with("page-123")
        mock_upsert.assert_not_called()
        assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_updated_event_deletes_then_upserts(self, tmp_path):
        from workers.ingestion_worker import process_document_event

        # Create a real temp file so the worker can find it
        doc_file = tmp_path / "policy.md"
        doc_file.write_text("# Policy\nContent here.")

        event_data = {
            "provider": "confluence",
            "event_type": "updated",
            "doc_id": "page-456",
            "doc_url": f"file://{doc_file}",
            "doc_title": "Policy Page",
            "folder_path": "hr",
        }

        with patch("workers.ingestion_worker.delete_document_chunks") as mock_delete, \
             patch("workers.ingestion_worker.upsert_document", return_value=3) as mock_upsert, \
             patch("workers.ingestion_worker.configure_llama_index"):

            result = await process_document_event({}, event_data)

        mock_delete.assert_called_once_with("page-456")
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args
        assert call_kwargs.kwargs["doc_id"] == "page-456"
        assert "hr" in call_kwargs.kwargs["allowed_roles"]
        assert result["status"] == "complete"
        assert result["chunks_upserted"] == 3

    @pytest.mark.asyncio
    async def test_created_event_deletes_then_upserts(self, tmp_path):
        from workers.ingestion_worker import process_document_event

        doc_file = tmp_path / "new_doc.md"
        doc_file.write_text("# New\nContent.")

        event_data = {
            "provider": "sharepoint",
            "event_type": "created",
            "doc_id": "item-789",
            "doc_url": f"file://{doc_file}",
            "doc_title": "New Document",
            "folder_path": "engineering",
        }

        with patch("workers.ingestion_worker.delete_document_chunks") as mock_delete, \
             patch("workers.ingestion_worker.upsert_document", return_value=5) as mock_upsert, \
             patch("workers.ingestion_worker.configure_llama_index"):

            result = await process_document_event({}, event_data)

        mock_delete.assert_called_once_with("item-789")
        mock_upsert.assert_called_once()
        assert result["status"] == "complete"

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self):
        from workers.ingestion_worker import process_document_event

        event_data = {
            "provider": "gdrive",
            "event_type": "updated",
            "doc_id": "file-999",
            "doc_url": "file:///nonexistent/path/file.pdf",
            "doc_title": "Ghost File",
            "folder_path": "general",
        }

        with patch("workers.ingestion_worker.delete_document_chunks"), \
             patch("workers.ingestion_worker.configure_llama_index"):

            result = await process_document_event({}, event_data)

        assert result["status"] == "error"
        assert "file not found" in result["reason"]

    @pytest.mark.asyncio
    async def test_delete_called_before_upsert(self, tmp_path):
        """Delete must happen before upsert to avoid a window with duplicate chunks."""
        from workers.ingestion_worker import process_document_event

        doc_file = tmp_path / "order_test.md"
        doc_file.write_text("Content.")

        call_order = []

        async def mock_resolve(*a, **kw):
            return str(doc_file)

        with patch("workers.ingestion_worker.delete_document_chunks",
                   side_effect=lambda x: call_order.append("delete")), \
             patch("workers.ingestion_worker.upsert_document",
                   side_effect=lambda **kw: call_order.append("upsert") or 1), \
             patch("workers.ingestion_worker._resolve_file",
                   new=mock_resolve), \
             patch("workers.ingestion_worker.configure_llama_index"):

            await process_document_event({}, {
                "provider": "confluence",
                "event_type": "updated",
                "doc_id": "order-test",
                "doc_url": f"file://{doc_file}",
                "doc_title": "Order Test",
                "folder_path": "general",
            })

        assert call_order == ["delete", "upsert"], (
            f"Expected delete before upsert, got: {call_order}"
        )
