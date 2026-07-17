"""Structured logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output. When set, a second
            DEBUG-level file sink scoped to ``src.llm.*`` is also created
            alongside it (``<stem>_llm_debug.log``) to capture full
            exception traces from the LLM client / endpoint pool — this
            is the diagnostic channel for diagnosing endpoint-death
            root causes (rate limit, empty content, timeout, etc.).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "{message}",
    )
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="50 MB",
            retention="10 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
        # Dedicated DEBUG sink for LLM diagnostics. Captures the full
        # exception chain (type / status_code / message / response id /
        # usage breakdown) so endpoint-death investigations no longer
        # depend on terminal scrollback.
        llm_debug_path = Path(log_file).with_name(
            Path(log_file).stem + "_llm_debug.log"
        )
        logger.add(
            str(llm_debug_path),
            level="DEBUG",
            rotation="50 MB",
            retention="30 days",
            filter=lambda record: record["name"].startswith("src.llm."),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )
