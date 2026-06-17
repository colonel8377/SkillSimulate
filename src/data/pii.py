"""PII scrubbing for data loaders (outline §7.5).

Implements the ethics-claim preprocessing that §7.5 line 444 specifies:
"用户名、IP 引用、个人链接等 PII 在预处理阶段移除". Without this module the
data loaders preserve raw platform handles and any PII embedded in message
text, contradicting the paper's stated ethics procedure.

Two layers of PII handling:

1. **Text-level stripping** — removes IP addresses, email addresses, URLs
   (especially personal links: personal sites, social profiles), and
   obvious username self-references from message text. Replaces them with
   neutral tokens so the surrounding sentence still parses for downstream
   NLP features.

2. **Identity pseudonymization** — replaces raw platform user IDs with
   deterministic salted hashes. The hash is stable within a run (so the
   interaction graph is preserved) but irreversible without the salt.

The default salt is fixed in code so artifacts are reproducible; rotate it
via the ``CADP_PII_SALT`` environment variable for stronger unlinkability
when publishing artifacts.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace

from src.data.schemas import Message, Thread


# Default salt — override via env var for stronger unlinkability on release
_DEFAULT_SALT = "cadp-pii-v1-2026"
_SALT = os.environ.get("CADP_PII_SALT", _DEFAULT_SALT)


# ---------------------------------------------------------------------------
# Text-level PII patterns
# ---------------------------------------------------------------------------

# IPv4 (deliberately simple — sufficient for talk-page / comment text)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# IPv6 — Wikipedia talk-page edit summaries and GitHub issue metadata
# frequently carry IPv6 addresses (anonymous editors / corporate egress).
# Covers full, compressed (``::``), and IPv4-mapped forms per RFC 4291.
_IPV6_RE = re.compile(
    r"(?:(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}"               # full
    r"|(?:[A-Fa-f0-9]{1,4}:){1,7}:"                               # trailing ::
    r"|(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}"               # compressed
    r"|(?:[A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}"
    r"|(?:[A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}"
    r"|[A-Fa-f0-9]{1,4}:(?:(?::[A-Fa-f0-9]{1,4}){1,6})"
    r"|::(?:[A-Fa-f0-9]{1,4}:){0,6}[A-Fa-f0-9]{1,4}"              # leading ::
    r"|::)"
    r"(?:/\d{1,3})?"                                               # optional CIDR
)

# Email addresses
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# URLs — covers http(s) and www. forms. We deliberately capture the whole
# URL because "personal links" (§7.5) are often social-media profiles or
# personal sites that uniquely identify a user.
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"]+[^\s<>\".,;!?)\]]"
)

# @-mentions (platform handles: @username, u/username)
_MENTION_RE = re.compile(r"(?:^|\s)(@[A-Za-z0-9_]{3,}|u/[A-Za-z0-9_]{3,})")

# Bare-handle username self-references — prose like "as UserX said" or
# "pinged Colonel83" without a sigil. Outline §7.5 specifies "usernames"
# as PII, which is broader than only sigil-prefixed mentions. We match
# the capitalized ``Word`` + optional trailing digits form (``UserX``,
# ``Colonel83``) which is the typical bare-handle shape; this is
# intentionally conservative to avoid scrubbing ordinary capitalized words.
# (G12 — PII scope expansion.)
_BARE_HANDLE_RE = re.compile(
    r"(?<=\s)([A-Z][a-zA-Z]{2,}[0-9]{1,4})(?=[\s.,;:!?)\"]|$)"
)

# Long digit runs that look like phone numbers or account IDs (>=7 digits)
_LONG_DIGIT_RE = re.compile(r"\b\d{7,}\b")


_TEXT_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EMAIL_RE, "[EMAIL]"),
    (_URL_RE, "[URL]"),
    (_IPV4_RE, "[IP]"),
    (_IPV6_RE, "[IP]"),
    # Mentions kept short — collapse handle but preserve that a mention occurred
    (_MENTION_RE, r" \g<1>".replace(r"\g<1>", "[MENTION]")),
    (_BARE_HANDLE_RE, "[MENTION]"),
    (_LONG_DIGIT_RE, "[ID]"),
)


def strip_pii_from_text(text: str) -> str:
    """Strip PII from a message text, replacing with neutral tokens.

    Args:
        text: Raw message text.

    Returns:
        Text with emails / URLs / IPs / mentions / long-digit IDs replaced
        by bracketed tokens ([EMAIL], [URL], [IP], [MENTION], [ID]).
    """
    if not text:
        return text
    scrubbed = text
    for pattern, repl in _TEXT_REPLACEMENTS:
        scrubbed = pattern.sub(repl, scrubbed)
    return scrubbed


# ---------------------------------------------------------------------------
# Identity pseudonymization
# ---------------------------------------------------------------------------


def anonymize_user_id(raw_id: str, salt: str | None = None) -> str:
    """Deterministic salted-hash pseudonymization of a user ID.

    The same raw_id always maps to the same pseudonym within a run
    (so interaction graphs survive pseudonymization) but the mapping is
    one-way without the salt.

    Args:
        raw_id: Raw platform user identifier.
        salt: Salt for the hash. Defaults to the module-level salt
            (env-configurable via ``CADP_PII_SALT``).

    Returns:
        Pseudonymous ID of the form ``"user_"`` + 16 hex chars.
    """
    if not raw_id:
        return "user_empty"
    s = salt if salt is not None else _SALT
    digest = hashlib.sha256(f"{s}::{raw_id}".encode("utf-8")).hexdigest()
    return f"user_{digest[:16]}"


# ---------------------------------------------------------------------------
# Thread-level application
# ---------------------------------------------------------------------------


def scrub_message(msg: Message, *, salt: str | None = None) -> Message:
    """Return a new ``Message`` with PII stripped + user_id pseudonymized.

    The original ``Message`` is not mutated (dataclass ``replace``).
    """
    new_text = strip_pii_from_text(msg.text)
    new_user_id = anonymize_user_id(msg.user_id, salt=salt)
    return replace(
        msg,
        text=new_text,
        user_id=new_user_id,
        # Preserve the original metadata but flag that scrubbing ran
        metadata={**msg.metadata, "_pii_scrubbed": True},
    )


def scrub_threads(
    threads: list[Thread],
    *,
    salt: str | None = None,
    skip: bool = False,
) -> list[Thread]:
    """Apply PII scrubbing to all messages in a list of threads.

    Args:
        threads: Loaded Thread objects with raw text + raw user_ids.
        salt: Optional salt override (env var ``CADP_PII_SALT`` used by default).
        skip: If True, return threads unchanged (escape hatch for unit tests
            that need raw data). Default False.

    Returns:
        New list of Thread objects (originals not mutated) with scrubbed
        messages and pseudonymized user_ids. ``Thread.participants`` is
        rebuilt from the pseudonymized user_ids.
    """
    if skip:
        return threads

    scrubbed_threads: list[Thread] = []
    for thread in threads:
        new_messages = [scrub_message(m, salt=salt) for m in thread.messages]
        new_thread = Thread(
            thread_id=thread.thread_id,
            platform=thread.platform,
            topic=thread.topic,
            messages=new_messages,
            participants={m.user_id for m in new_messages},
        )
        scrubbed_threads.append(new_thread)
    return scrubbed_threads
