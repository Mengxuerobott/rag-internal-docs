"""
webhooks/router.py
──────────────────
FastAPI router that receives webhook POST requests from document management
systems, verifies their signatures, normalises their payloads into a
canonical WebhookEvent schema, and enqueues the corresponding ingestion job
in Redis via the ARQ job queue.

Supported providers:
  POST /webhooks/confluence  — Confluence Server / Cloud page events
  POST /webhooks/sharepoint  — SharePoint Online list/file change notifications
  POST /webhooks/gdrive      — Google Drive push notifications

The router deliberately does nothing except:
  1. Verify the signature (authenticate the sender).
  2. Parse the payload into a WebhookEvent.
  3. Enqueue a job and return 200 immediately.

All heavy work (LlamaParse, embedding, Qdrant upsert) happens in the ARQ
worker process. This keeps webhook response time under 100ms — most providers
retry if you don't acknowledge within their timeout window (Confluence: 10s,
SharePoint: 5s, Google Drive: 30s).

Event-driven flow diagram:
  DMS change → POST /webhooks/<provider>
                  │  verify HMAC
                  │  normalise payload → WebhookEvent
                  │  arq.enqueue_job("process_document_event", event)
                  └──→ 200 {"status": "queued", "job_id": "..."}

                        Redis queue
                              │
                    ARQ worker picks up job
                              │
                    delete_document_chunks(doc_id)   [always]
                              │
                    upsert_document(file_path, ...)  [if event_type != "deleted"]
"""

import json
from enum import Enum
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from webhooks.signature import (
    verify_confluence_signature,
    verify_gdrive_signature,
    verify_sharepoint_signature,
)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ── Canonical event model ─────────────────────────────────────────────────────

class EventType(str, Enum):
    CREATED  = "created"
    UPDATED  = "updated"
    DELETED  = "deleted"


class WebhookEvent(BaseModel):
    """
    Provider-agnostic representation of a document change event.
    All provider-specific fields are normalised into this schema before
    being enqueued so the worker never needs to know which DMS fired the event.
    """
    provider:   str        = Field(..., description="confluence | sharepoint | gdrive")
    event_type: EventType  = Field(..., description="created | updated | deleted")
    doc_id:     str        = Field(..., description="Stable DMS document ID")
    doc_title:  str        = Field(default="", description="Human-readable title")
    doc_url:    str        = Field(default="", description="Link to the document in the DMS")
    folder_path: str       = Field(default="general", description="Parent folder / space / library")
    raw_payload: dict      = Field(default_factory=dict, description="Original provider payload (for debugging)")


# ── ARQ pool helper ───────────────────────────────────────────────────────────

async def _get_redis_pool():
    """
    Return an ARQ Redis pool parsed from settings.REDIS_URL.
    ARQ uses its own RedisSettings model (not the raw URL string).
    """
    from urllib.parse import urlparse
    parsed = urlparse(settings.REDIS_URL)
    return await create_pool(
        RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            database=int(parsed.path.lstrip("/") or 0),
        )
    )


async def _enqueue(event: WebhookEvent) -> str:
    """
    Enqueue a document ingestion job in Redis and return the job ID.
    The ARQ worker (workers/ingestion_worker.py) picks this up asynchronously.
    """
    redis = await _get_redis_pool()
    job = await redis.enqueue_job(
        "process_document_event",
        event.model_dump(),                 # serialise to plain dict for Redis
        _job_id=f"{event.provider}:{event.doc_id}",  # idempotency key
        _job_try=1,
    )
    await redis.aclose()
    job_id = job.job_id if job else f"{event.provider}:{event.doc_id}"
    logger.info(
        f"Enqueued job {job_id!r} — "
        f"provider={event.provider!r} "
        f"event_type={event.event_type!r} "
        f"doc_id={event.doc_id!r}"
    )
    return job_id


# ── Confluence webhook ────────────────────────────────────────────────────────

@router.post("/confluence")
async def confluence_webhook(request: Request) -> dict:
    """
    Receive a Confluence page event.

    Confluence sends events for: page_created, page_updated, page_removed,
    blog_post_created, blog_post_updated, blog_post_removed.

    Configure in Confluence: Settings → System → Webhooks → Create webhook
    Set URL to: https://your-api.example.com/webhooks/confluence
    Set Secret to match WEBHOOK_SECRET_CONFLUENCE in .env.

    Example payload (abbreviated):
    {
      "event": "page_updated",
      "page": {
        "id": "12345",
        "title": "Engineering Onboarding",
        "space": {"key": "ENG"},
        "_links": {"self": "https://..."}
      },
      "timestamp": 1700000000000
    }
    """
    await verify_confluence_signature(request, settings.WEBHOOK_SECRET_CONFLUENCE)

    body = await request.body()
    try:
        payload: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid JSON payload")

    # Normalise Confluence event → WebhookEvent
    event_str: str = payload.get("event", "")
    page: dict = payload.get("page", payload.get("blogpost", {}))

    # Map Confluence event name → EventType
    if "created" in event_str:
        event_type = EventType.CREATED
    elif "removed" in event_str or "deleted" in event_str:
        event_type = EventType.DELETED
    else:
        event_type = EventType.UPDATED

    doc_id    = str(page.get("id", "unknown"))
    doc_title = page.get("title", "")
    doc_url   = page.get("_links", {}).get("self", "")
    space_key = page.get("space", {}).get("key", "general").lower()

    event = WebhookEvent(
        provider="confluence",
        event_type=event_type,
        doc_id=doc_id,
        doc_title=doc_title,
        doc_url=doc_url,
        folder_path=space_key,
        raw_payload=payload,
    )

    job_id = await _enqueue(event)
    return {"status": "queued", "job_id": job_id, "event_type": event_type}


# ── SharePoint webhook ────────────────────────────────────────────────────────

@router.post("/sharepoint")
async def sharepoint_webhook(request: Request) -> dict:
    """
    Receive a SharePoint Online list/file change notification.

    SharePoint sends change notifications for list item creates, updates,
    and deletes. The notification body contains minimal metadata; full file
    details are fetched by the worker using the Graph API (doc_url).

    Configure via Microsoft Graph subscription API:
    POST https://graph.microsoft.com/v1.0/subscriptions
    Set notificationUrl to: https://your-api.example.com/webhooks/sharepoint
    Set clientState to match WEBHOOK_SECRET_SHAREPOINT in .env.

    Note: SharePoint sends a validationToken query param on first subscription.
    This endpoint returns it as plain text to complete the handshake.

    Example payload:
    {
      "value": [{
        "subscriptionId": "abc",
        "clientState": "<your-secret>",
        "changeType": "updated",
        "resource": "sites/site-id/lists/list-id/items/item-id",
        "resourceData": {"id": "item-id", "@odata.type": "#Microsoft.Graph.ListItem"}
      }]
    }
    """
    # SharePoint subscription validation handshake
    validation_token = request.query_params.get("validationToken")
    if validation_token:
        logger.info("SharePoint subscription validation handshake received")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=validation_token, media_type="text/plain")

    await verify_sharepoint_signature(request, settings.WEBHOOK_SECRET_SHAREPOINT)

    body = await request.body()
    try:
        payload: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid JSON payload")

    notifications: list[dict] = payload.get("value", [payload])
    job_ids = []

    for notif in notifications:
        change_type: str = notif.get("changeType", "updated").lower()
        resource: str = notif.get("resource", "")

        if change_type == "deleted":
            event_type = EventType.DELETED
        elif change_type == "created":
            event_type = EventType.CREATED
        else:
            event_type = EventType.UPDATED

        # Extract item ID from resource path: "sites/.../lists/.../items/ITEM_ID"
        doc_id = resource.rstrip("/").split("/")[-1] if resource else "unknown"

        # Derive folder from list name in resource path (heuristic)
        parts = resource.split("/")
        folder = "general"
        if "lists" in parts:
            idx = parts.index("lists")
            folder = parts[idx + 1] if idx + 1 < len(parts) else "general"

        resource_data: dict = notif.get("resourceData", {})
        event = WebhookEvent(
            provider="sharepoint",
            event_type=event_type,
            doc_id=doc_id,
            doc_title=resource_data.get("name", ""),
            doc_url=resource,
            folder_path=folder.lower(),
            raw_payload=notif,
        )
        job_id = await _enqueue(event)
        job_ids.append(job_id)

    return {"status": "queued", "job_ids": job_ids, "count": len(job_ids)}


# ── Google Drive webhook ──────────────────────────────────────────────────────

@router.post("/gdrive")
async def gdrive_webhook(request: Request) -> dict:
    """
    Receive a Google Drive push notification.

    Google Drive uses a push-notification model: when a watched file changes,
    Drive sends a POST to your webhook URL with the change info in HTTP headers
    (not the body — the body is empty for change notifications).

    Register a watch channel via:
    POST https://www.googleapis.com/drive/v3/files/{fileId}/watch
    Set address to: https://your-api.example.com/webhooks/gdrive
    Set token to match WEBHOOK_SECRET_GDRIVE in .env.

    Relevant request headers sent by Google:
      X-Goog-Channel-ID:      <channel-id you set>
      X-Goog-Resource-ID:     <stable file/folder ID>
      X-Goog-Resource-State:  sync | update | add | remove | trash | untrash | change
      X-Goog-Changed:         content,properties   (what changed)
      X-Goog-Channel-Token:   <your secret token>
      X-Goog-Message-Number:  sequential message counter (for deduplication)
    """
    await verify_gdrive_signature(request, settings.WEBHOOK_SECRET_GDRIVE)

    resource_state = request.headers.get("X-Goog-Resource-State", "")
    resource_id    = request.headers.get("X-Goog-Resource-ID", "unknown")
    channel_id     = request.headers.get("X-Goog-Channel-ID", "")
    changed_fields = request.headers.get("X-Goog-Changed", "")
    msg_number     = request.headers.get("X-Goog-Message-Number", "0")

    logger.debug(
        f"GDrive notification: state={resource_state!r} "
        f"resource_id={resource_id!r} msg={msg_number}"
    )

    # "sync" is the initial handshake message — acknowledge and return
    if resource_state == "sync":
        logger.info(f"GDrive channel sync handshake for channel_id={channel_id!r}")
        return {"status": "ok", "message": "sync acknowledged"}

    # Map Drive resource state → EventType
    if resource_state in ("remove", "trash"):
        event_type = EventType.DELETED
    elif resource_state == "add":
        event_type = EventType.CREATED
    else:
        event_type = EventType.UPDATED

    # Google Drive doesn't include the filename or parent folder in the
    # notification headers — the worker must fetch those via the Drive API
    # using the resource_id. We store resource_id as doc_id and doc_url.
    event = WebhookEvent(
        provider="gdrive",
        event_type=event_type,
        doc_id=resource_id,
        doc_title="",         # worker resolves via Drive API
        doc_url=f"https://drive.google.com/file/d/{resource_id}",
        folder_path="general",  # worker resolves via Drive API
        raw_payload={
            "resource_state": resource_state,
            "resource_id": resource_id,
            "channel_id": channel_id,
            "changed": changed_fields,
            "message_number": msg_number,
        },
    )

    job_id = await _enqueue(event)
    return {"status": "queued", "job_id": job_id, "event_type": event_type}


# ── Job status endpoint ───────────────────────────────────────────────────────

@router.get("/job/{job_id}")
async def job_status(job_id: str) -> dict:
    """
    Check the status of an enqueued ingestion job.

    Returns one of: queued | in_progress | complete | failed | not_found
    """
    from arq.jobs import Job, JobStatus

    redis = await _get_redis_pool()
    job = Job(job_id, redis)

    try:
        job_status_val = await job.status()
        result = None
        if job_status_val == JobStatus.complete:
            result_obj = await job.result(timeout=0)
            result = str(result_obj) if result_obj is not None else None

        return {
            "job_id": job_id,
            "status": job_status_val.value,
            "result": result,
        }
    except Exception as e:
        return {"job_id": job_id, "status": "not_found", "error": str(e)}
    finally:
        await redis.aclose()
