"""Smoke test for the GPU-aware embedder backend.

Loads the shared and SIP embedders, prints device/dim, and verifies both
synchronous batch encode and async executor offload.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make repo root importable when running from scripts/
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config.embedder import run_embed_in_executor
from src.config.settings import get_shared_embedder, get_sip_embedder


def test_sync() -> None:
    shared = get_shared_embedder()
    sip = get_sip_embedder()
    print(
        f"Shared: model={shared.model_name}, "
        f"dim={shared.get_sentence_embedding_dimension()}, device={shared.device}"
    )
    print(
        f"SIP:    model={sip.model_name}, "
        f"dim={sip.get_sentence_embedding_dimension()}, device={sip.device}"
    )

    texts = [
        "hello world",
        "this is a test sentence for the gpu embedder",
    ]
    emb = shared.encode(texts)
    assert emb.shape == (len(texts), shared.get_sentence_embedding_dimension())
    print("Sync batch encode OK")


async def test_async() -> None:
    shared = get_shared_embedder()
    emb = await run_embed_in_executor(
        shared.encode, "async single text", show_progress_bar=False
    )
    assert emb.shape == (shared.get_sentence_embedding_dimension(),)
    print("Async single encode OK")


async def main() -> None:
    test_sync()
    await test_async()
    print("All embedder smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
