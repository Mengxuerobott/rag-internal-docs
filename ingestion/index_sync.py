"""
ingestion/index_sync.py
───────────────────────
Durable storage for the LlamaIndex persist directory.

The problem
───────────
Chunk vectors live in Qdrant and survive restarts. The *docstore* does not:
`storage_context.persist()` writes it to INDEX_PERSIST_DIR on the local
filesystem, which on Fargate is destroyed with the task. So on every deploy
`get_or_build_index()` finds an empty directory and re-runs the whole ingestion
pipeline — re-embedding every leaf node through the OpenAI API — just to
reconstruct state that was already computed.

That matters more than it looks, because the docstore is not a cache. Only leaf
nodes are embedded into Qdrant; the parent nodes exist *solely* in the docstore,
and AutoMergingRetriever walks leaf → parent through it. Lose it and merging
silently stops working.

Rebuilding is also not a matter of re-chunking: HierarchicalNodeParser assigns
random node IDs, so a rebuilt docstore would not match the IDs already stored in
Qdrant and every parent pointer would dangle. That is why the current code
rebuilds *both* together — correct, but it pays the full embedding cost on every
restart, and now that CI deploys on each push to main, on every commit.

The fix
───────
Mirror the persist directory to S3, so the docstore and the vectors it refers to
survive together. Download it at startup, upload it after any ingestion that
changes it.

Optional by design: with INDEX_S3_BUCKET unset every function here is a no-op
and the behaviour is exactly what it was — local disk, rebuild when empty. That
keeps local development and the test suite free of any AWS dependency.

Failure policy
──────────────
Never raise. A missing bucket, expired credentials or a network blip degrades to
"rebuild locally", which is slow and costs embedding spend but is still correct.
Taking the API down because object storage was briefly unavailable would be a
worse trade.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from config import settings


def s3_configured() -> bool:
    """True when an S3 bucket has been configured for the index store."""
    return bool(settings.INDEX_S3_BUCKET)


def _client():
    """Build an S3 client, or None if boto3 is unavailable."""
    try:
        import boto3

        return boto3.client("s3", region_name=settings.AWS_REGION or None)
    except Exception as e:
        logger.warning(f"Could not create an S3 client ({e}) — index sync disabled.")
        return None


def _key_for(local_path: Path, root: Path) -> str:
    """Map a local file to its S3 key, preserving the directory layout."""
    relative = local_path.relative_to(root).as_posix()
    prefix = settings.INDEX_S3_PREFIX.strip("/")
    return f"{prefix}/{relative}" if prefix else relative


def download_index_store(dest_dir: str | None = None) -> bool:
    """
    Fetch the persisted index store from S3 into `dest_dir`.

    Returns True only if at least one file was downloaded, i.e. the caller can
    now load the index instead of rebuilding it. Never raises.
    """
    if not s3_configured():
        return False

    dest = Path(dest_dir or settings.INDEX_PERSIST_DIR)
    client = _client()
    if client is None:
        return False

    prefix = settings.INDEX_S3_PREFIX.strip("/")
    bucket = settings.INDEX_S3_BUCKET

    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        downloaded = 0
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Skip "directory" markers some tools create.
                if key.endswith("/"):
                    continue

                relative = key[len(prefix):].lstrip("/") if prefix else key
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(target))
                downloaded += 1

        if downloaded:
            logger.info(
                f"Index store restored from s3://{bucket}/{prefix} "
                f"({downloaded} file(s)) — skipping re-ingestion."
            )
            return True

        logger.info(
            f"No index store at s3://{bucket}/{prefix} — will build it and upload."
        )
        return False

    except Exception as e:
        logger.warning(
            f"Could not restore the index store from S3 ({e}) — "
            f"falling back to rebuilding it locally."
        )
        return False


def upload_index_store(src_dir: str | None = None) -> bool:
    """
    Mirror `src_dir` to S3 so the next task can restore it.

    Call after any ingestion that rewrites the persist directory — a full build
    or an incremental upsert. Returns True on success. Never raises.
    """
    if not s3_configured():
        return False

    src = Path(src_dir or settings.INDEX_PERSIST_DIR)
    if not src.exists():
        logger.warning(f"Nothing to upload — {src} does not exist.")
        return False

    client = _client()
    if client is None:
        return False

    bucket = settings.INDEX_S3_BUCKET

    try:
        uploaded = 0
        for path in src.rglob("*"):
            if path.is_file():
                client.upload_file(str(path), bucket, _key_for(path, src))
                uploaded += 1

        logger.info(
            f"Index store uploaded to "
            f"s3://{bucket}/{settings.INDEX_S3_PREFIX.strip('/')} "
            f"({uploaded} file(s))."
        )
        return True

    except Exception as e:
        # The index is still correct in memory and on local disk; only the next
        # restart pays for this, by rebuilding.
        logger.warning(f"Could not upload the index store to S3 ({e}).")
        return False


def local_index_store_present(path: str | None = None) -> bool:
    """True if a non-empty persist directory already exists locally."""
    target = path or settings.INDEX_PERSIST_DIR
    return os.path.exists(target) and bool(os.listdir(target))
