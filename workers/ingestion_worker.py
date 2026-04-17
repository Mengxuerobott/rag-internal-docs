"""
workers/ingestion_worker.py
────────────────────────────
ARQ async worker that processes document ingestion jobs enqueued by the
webhook router.

Responsibilities:
  1. On document CREATED or UPDATED:
       a. Delete existing Qdrant chunks for that doc_id (idempotent — safe
          if the document was never indexed before).
       b. Download / locate the updated file.
       c. Re-parse with LlamaParse, chunk, embed, and upsert into Qdrant.

  2. On document DELETED:
       a. Delete all Qdrant chunks for that doc_id.
       No re-ingestion.

  3. Retry failed jobs up to WORKER_MAX_RETRIES times with exponential backoff.

Why ARQ?
─────────
ARQ is a lightweight async Redis-backed job queue built for asyncio.
It requires only a running Redis instance (no Celery broker, no RabbitMQ).
Each worker is a single Python process; scale by running more replicas
(see docker-compose.yml `--scale worker=N`).

Run the worker locally:
    python -m workers.ingestion_worker

Or via Docker Compose:
    docker compose up worker

Architecture note — why a separate worker process?
────────────────────────────────────────────────────
The webhook endpoint must respond within ~5 seconds or the provider will
retry (some providers retry aggressively, causing duplicate jobs). Embedding
a 50-page PDF can take 30-120 seconds. Running that in the FastAPI request
handler would block the entire event loop.

The worker is a separate process: the API acknowledges the webhook instantly,
the worker does the heavy lifting asynchronously. Redis is the durable buffer
between them — jobs survive worker restarts.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import aiohttp
from arq import Worker
from arq.connections import RedisSettings
from loguru import logger
from urllib.parse import urlparse

from config import settings
from ingestion.embedder import configure_llama_index, delete_document_chunks, upsert_document
from auth.rbac import get_allowed_roles_for_path


# ── Job functions ─────────────────────────────────────────────────────────────

async def process_document_event(ctx: dict, event_data: dict) -> dict:
    """
    Main job function — called by ARQ for every enqueued webhook event.

    Args:
        ctx:         ARQ worker context (contains redis pool, job metadata).
        event_data:  Serialised WebhookEvent dict from webhooks/router.py.

    Returns:
        Result dict stored in Redis for status polling via GET /webhooks/job/{id}.
    """
    provider    = event_data["provider"]
    event_type  = event_data["event_type"]
    doc_id      = event_data["doc_id"]
    doc_url     = event_data.get("doc_url", "")
    doc_title   = event_data.get("doc_title", "unknown")
    folder_path = event_data.get("folder_path", "general")

    logger.info(
        f"Processing job — provider={provider!r} "
        f"event_type={event_type!r} "
        f"doc_id={doc_id!r} "
        f"title={doc_title!r}"
    )

    # ── Step 1: Always delete stale chunks first ──────────────────────────────
    # This is safe even if the document was never ingested (returns 0 deletes).
    # For updates: clears old chunks so we don't accumulate duplicates.
    # For deletes: this is the only step.
    logger.info(f"Deleting stale chunks for doc_id={doc_id!r}")
    delete_document_chunks(doc_id)

    if event_type == "deleted":
        logger.info(f"Document deleted — chunks removed, no re-ingestion needed")
        return {"status": "deleted", "doc_id": doc_id}

    # ── Step 2: Download or locate the file ──────────────────────────────────
    file_path = await _resolve_file(provider, doc_id, doc_url, event_data)
    if file_path is None:
        logger.error(
            f"Could not resolve file for doc_id={doc_id!r} — "
            f"provider={provider!r} doc_url={doc_url!r}"
        )
        return {"status": "error", "doc_id": doc_id, "reason": "file not found"}

    # ── Step 3: Determine RBAC roles from folder path ─────────────────────────
    allowed_roles = get_allowed_roles_for_path(folder_path)
    logger.debug(f"RBAC: folder={folder_path!r} → allowed_roles={allowed_roles}")

    # ── Step 4: Re-ingest ─────────────────────────────────────────────────────
    configure_llama_index()
    n_chunks = upsert_document(
        file_path=file_path,
        doc_id=doc_id,
        allowed_roles=allowed_roles,
    )

    # Clean up temp file if we downloaded it
    if file_path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(file_path)
        except OSError:
            pass

    logger.info(
        f"Ingestion complete — doc_id={doc_id!r} "
        f"chunks={n_chunks} event_type={event_type!r}"
    )
    return {
        "status": "complete",
        "doc_id": doc_id,
        "chunks_upserted": n_chunks,
        "event_type": event_type,
    }


async def _resolve_file(
    provider: str,
    doc_id: str,
    doc_url: str,
    event_data: dict,
) -> str | None:
    """
    Return a local file path for the document, downloading it if necessary.

    For local files (batch ingestion test): doc_url starts with "file://"
    For remote DMS files: download via authenticated HTTP to a temp file.

    In a real deployment you would:
      - Confluence: use the Confluence REST API with a service-account token.
      - SharePoint: use the Microsoft Graph API with app credentials.
      - Google Drive: use the Drive API with a service-account key file.

    This implementation handles the local-file and generic-HTTP cases and
    provides clear extension points for each DMS.
    """
    if not doc_url:
        logger.warning(f"No doc_url for doc_id={doc_id!r} — cannot download")
        return None

    # ── Local file (for testing without a real DMS) ───────────────────────────
    if doc_url.startswith("file://"):
        local_path = doc_url[len("file://"):]
        if os.path.exists(local_path):
            return local_path
        logger.error(f"Local file not found: {local_path!r}")
        return None

    # ── Confluence export URL ─────────────────────────────────────────────────
    if provider == "confluence":
        return await _download_confluence(doc_id, doc_url, event_data)

    # ── SharePoint download URL ───────────────────────────────────────────────
    if provider == "sharepoint":
        return await _download_sharepoint(doc_id, doc_url, event_data)

    # ── Google Drive export URL ───────────────────────────────────────────────
    if provider == "gdrive":
        return await _download_gdrive(doc_id)

    # ── Generic HTTP fallback ─────────────────────────────────────────────────
    return await _download_url(doc_url, suffix=".pdf")


async def _download_url(url: str, suffix: str = ".bin") -> str | None:
    """Download a URL to a named temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    logger.error(f"Download failed: HTTP {resp.status} for {url!r}")
                    return None
                tmp.write(await resp.read())
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"Download error for {url!r}: {e}")
        tmp.close()
        os.unlink(tmp.name)
        return None


async def _download_confluence(doc_id: str, doc_url: str, event_data: dict) -> str | None:
    """
    Download a Confluence page as PDF using the Confluence REST API.

    Requires environment variables:
        CONFLUENCE_BASE_URL  — e.g. https://yourcompany.atlassian.net/wiki
        CONFLUENCE_API_TOKEN — Atlassian API token (personal access token)
        CONFLUENCE_EMAIL     — Account email associated with the API token
    """
    base_url  = os.getenv("CONFLUENCE_BASE_URL", "").rstrip("/")
    api_token = os.getenv("CONFLUENCE_API_TOKEN", "")
    email     = os.getenv("CONFLUENCE_EMAIL", "")

    if not all([base_url, api_token, email]):
        logger.warning(
            "CONFLUENCE_BASE_URL / CONFLUENCE_API_TOKEN / CONFLUENCE_EMAIL "
            "not set — cannot download Confluence document. "
            "Using doc_url as direct download link instead."
        )
        return await _download_url(doc_url, suffix=".pdf")

    import base64
    creds = base64.b64encode(f"{email}:{api_token}".encode()).decode()
    export_url = f"{base_url}/rest/api/content/{doc_id}/pdf"
    headers = {"Authorization": f"Basic {creds}"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(export_url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    logger.error(f"Confluence PDF export failed: HTTP {resp.status}")
                    return None
                tmp.write(await resp.read())
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"Confluence download error: {e}")
        tmp.close()
        os.unlink(tmp.name)
        return None


async def _download_sharepoint(doc_id: str, doc_url: str, event_data: dict) -> str | None:
    """
    Download a SharePoint file using the Microsoft Graph API.

    Requires environment variables:
        SHAREPOINT_TENANT_ID  — Azure AD tenant ID
        SHAREPOINT_CLIENT_ID  — App registration client ID
        SHAREPOINT_CLIENT_SECRET — App registration client secret
    """
    tenant_id     = os.getenv("SHAREPOINT_TENANT_ID", "")
    client_id     = os.getenv("SHAREPOINT_CLIENT_ID", "")
    client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET", "")

    if not all([tenant_id, client_id, client_secret]):
        logger.warning(
            "SHAREPOINT_* credentials not set — cannot download SharePoint document."
        )
        return None

    # Step 1: Acquire OAuth2 token via client credentials flow
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }) as resp:
            if resp.status != 200:
                logger.error(f"SharePoint OAuth2 token request failed: HTTP {resp.status}")
                return None
            token_data = await resp.json()
            access_token = token_data.get("access_token")

    # Step 2: Download file content via Graph API
    # resource path format: "sites/{site-id}/lists/{list-id}/items/{item-id}"
    download_url = f"https://graph.microsoft.com/v1.0/{doc_url.lstrip('/')}/driveItem/content"
    headers = {"Authorization": f"Bearer {access_token}"}

    return await _download_url(download_url, suffix=".docx")  # SharePoint files are usually DOCX


async def _download_gdrive(doc_id: str) -> str | None:
    """
    Download a Google Drive file using the Drive API.

    Requires environment variable:
        GOOGLE_SERVICE_ACCOUNT_JSON — Path to service account key file
    """
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not sa_path or not os.path.exists(sa_path):
        logger.warning(
            "GOOGLE_SERVICE_ACCOUNT_JSON not set or file not found — "
            "cannot download Google Drive document."
        )
        return None

    # Use google-auth library (install: pip install google-auth aiohttp)
    try:
        import json as _json
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GoogleRequest

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        with open(sa_path) as f:
            sa_info = _json.load(f)
        credentials = service_account.Credentials.from_service_account_info(
            sa_info, scopes=scopes
        )
        credentials.refresh(GoogleRequest())
        access_token = credentials.token
    except Exception as e:
        logger.error(f"Google service account auth failed: {e}")
        return None

    # Export as PDF (works for Docs/Sheets/Slides; for binary files use ?alt=media)
    export_url = (
        f"https://www.googleapis.com/drive/v3/files/{doc_id}/export"
        "?mimeType=application/pdf"
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(export_url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 400:
                    # Binary file — download raw instead of exporting
                    raw_url = f"https://www.googleapis.com/drive/v3/files/{doc_id}?alt=media"
                    async with session.get(raw_url, timeout=aiohttp.ClientTimeout(total=120)) as r2:
                        tmp.write(await r2.read())
                elif resp.status == 200:
                    tmp.write(await resp.read())
                else:
                    logger.error(f"GDrive download failed: HTTP {resp.status}")
                    return None
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"GDrive download error: {e}")
        tmp.close()
        os.unlink(tmp.name)
        return None


# ── Worker startup / shutdown hooks ──────────────────────────────────────────

async def startup(ctx: dict) -> None:
    """Called once when the worker process starts."""
    logger.info("ARQ ingestion worker starting up")
    configure_llama_index()
    logger.info("LlamaIndex configured — worker ready")


async def shutdown(ctx: dict) -> None:
    """Called once when the worker process shuts down."""
    logger.info("ARQ ingestion worker shutting down")


# ── Worker configuration ──────────────────────────────────────────────────────

class WorkerSettings:
    """
    ARQ worker settings class.
    ARQ discovers this by convention when you run `arq workers.ingestion_worker.WorkerSettings`.
    """
    functions = [process_document_event]
    on_startup = startup
    on_shutdown = shutdown

    # Redis connection
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

    # Concurrency: how many jobs this worker processes in parallel
    max_jobs = settings.WORKER_CONCURRENCY

    # Retry policy: jobs that raise an exception are retried up to max_tries
    max_tries = settings.WORKER_MAX_RETRIES
    retry_delay = settings.WORKER_RETRY_DELAY_S  # seconds between retries

    # Keep completed job results in Redis for 24 hours (for status polling)
    keep_result = 86_400

    # Job timeout: kill jobs that run longer than this (in seconds)
    job_timeout = 600   # 10 minutes — generous for large PDFs


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run with:
        python -m workers.ingestion_worker

    Or using the ARQ CLI (recommended for production — handles signals cleanly):
        arq workers.ingestion_worker.WorkerSettings
    """
    import sys
    from arq import run_worker

    logger.info(f"Starting ARQ worker — Redis: {settings.REDIS_URL}")
    run_worker(WorkerSettings)
