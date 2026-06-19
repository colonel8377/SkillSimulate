"""GPU-aware embedder wrapper with async executor offloading."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from loguru import logger


# One worker is enough: the GPU handles internal batch parallelism.
# Multiple workers would just contend for the same GPU.
_embed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cadp_embed_")


def run_embed_in_executor(fn, *args, **kwargs):
    """Run a sync embedding function in the shared thread pool."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_embed_executor, lambda: fn(*args, **kwargs))


def _resolve_device(device: str, gpu_id: int) -> str:
    """Resolve 'auto' / 'cpu' / 'cuda:N' to a concrete device string.

    Args:
        device: User-facing device spec. 'auto' selects CUDA if available.
        gpu_id: CUDA device index used when device='auto' and CUDA is available.

    Returns:
        Concrete device string such as 'cuda:0' or 'cpu'.
    """
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        return device
    if device == "auto":
        force_cpu = os.getenv("CADP_FORCE_CPU_EMBEDDER", "0") == "1"
        if force_cpu:
            logger.warning("CADP_FORCE_CPU_EMBEDDER=1 — embedders forced to CPU")
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{gpu_id}"
        except Exception:
            pass
        logger.warning(
            "CUDA not available or torch import failed — falling back to CPU embedder"
        )
        return "cpu"
    logger.warning(f"Unknown embedder device '{device}' — falling back to auto")
    return _resolve_device("auto", gpu_id)


class EmbedderWrapper:
    """Thin wrapper around SentenceTransformer with device/batch config."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        trust_remote_code = os.getenv("CADP_EMBEDDER_TRUST_REMOTE_CODE", "0") == "1"
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self.device = str(self.model.device)
        self.batch_size = batch_size
        logger.info(
            f"Loaded embedder '{model_name}' on {self.device}, batch_size={batch_size}"
        )

    def encode(self, texts, show_progress_bar: bool = False, **kwargs: Any) -> np.ndarray:
        """Encode texts, passing batch_size for multi-text inputs."""
        if isinstance(texts, list) and len(texts) > 1:
            kwargs.setdefault("batch_size", self.batch_size)
        return self.model.encode(texts, show_progress_bar=show_progress_bar, **kwargs)

    def get_sentence_embedding_dimension(self) -> int:
        # sentence-transformers >= 3.0 renamed the method; support both names.
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())
