"""Held-out event annotation protocol for Predictive Fidelity (outline §5.3).

Ground-truth coding protocol:
  - 2 annotators independently code controversial events
    (conflict escalation / persuasion success / consensus formation)
  - Inter-annotator agreement: Cohen's κ ≥ 0.7
  - Consensus label = agreed label; disagreements are flagged for
    adjudication and excluded from the ground-truth set until resolved

This module provides:
  - HeldOutEvent dataclass + JSONL load/save
  - Cohen's κ computation
  - LLM-based dual annotator (two independent prompts) for generating
    candidate annotations that a human then reviews
  - Consensus resolution + agreement reporting

The LLM dual annotator is a pipeline convenience; the outline's protocol
is human annotators. Treat its output as draft annotations requiring
human adjudication before use as ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger


# Event types coded on held-out threads (outline §4.1, §5.3)
# Outline §4.1 lists three event categories for the held-out annotation
# protocol: conflict escalation / persuasion success / consensus formation.
# Outline §5.3 enumerates the predictive-fidelity tasks as a subset:
# conflict / persuasion / escalation. EVENT_CONSENSUS is therefore defined
# for the annotation universe (annotators may code it) but is intentionally
# not consumed as a predictive-fidelity task in src/evaluation/predictive.py.
EVENT_CONFLICT = "conflict"            # does a conflict occur / who engages
EVENT_PERSUASION = "persuasion"        # does persuasion succeed (delta awarded)
EVENT_ESCALATION = "escalation"        # does conflict escalate
EVENT_CONSENSUS = "consensus"          # does consensus form (annotation-only)

ALL_EVENT_TYPES = (
    EVENT_CONFLICT,
    EVENT_PERSUASION,
    EVENT_ESCALATION,
    EVENT_CONSENSUS,
)

# Minimum Cohen's κ required before consensus labels are usable (outline §5.3)
KAPPA_THRESHOLD = 0.7


@dataclass
class HeldOutEvent:
    """One held-out event with two independent annotations."""

    event_id: str
    thread_id: str
    event_type: str                       # one of ALL_EVENT_TYPES
    annotator_1: int                      # 0 / 1 binary label
    annotator_2: int
    annotator_1_notes: str = ""
    annotator_2_notes: str = ""
    consensus_label: int | None = None    # set after agreement check
    needs_adjudication: bool = False

    def resolve_consensus(self) -> None:
        """Set consensus_label; flag disagreements for adjudication."""
        if self.annotator_1 == self.annotator_2:
            self.consensus_label = self.annotator_1
            self.needs_adjudication = False
        else:
            self.consensus_label = None
            self.needs_adjudication = True


@dataclass
class AgreementReport:
    """Inter-annotator agreement summary for one event type."""

    event_type: str
    n_events: int
    n_agreed: int
    n_disputed: int
    cohen_kappa: float
    meets_threshold: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Cohen's κ
# ----------------------------------------------------------------------

def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's κ for two parallel annotation lists.

    Args:
        labels_a: Annotations from annotator 1.
        labels_b: Annotations from annotator 2 (same length).

    Returns:
        Cohen's κ in [-1, 1]. Returns 0.0 for empty/degenerate input.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Annotation lists must be equal length: {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        return 0.0

    labels = sorted(set(labels_a) | set(labels_b))

    # Observed agreement
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement by chance
    expected = 0.0
    for lbl in labels:
        p_a = sum(1 for a in labels_a if a == lbl) / n
        p_b = sum(1 for b in labels_b if b == lbl) / n
        expected += p_a * p_b

    if expected >= 1.0:
        return 0.0  # degenerate (both annotators constant & identical)

    return (observed - expected) / (1.0 - expected)


# ----------------------------------------------------------------------
# Consensus resolution
# ----------------------------------------------------------------------

def resolve_events(events: list[HeldOutEvent]) -> dict[str, AgreementReport]:
    """Resolve consensus labels and compute per-event-type agreement.

    Returns:
        Dict event_type -> AgreementReport.
    """
    reports: dict[str, AgreementReport] = {}

    for event_type in ALL_EVENT_TYPES:
        subset = [e for e in events if e.event_type == event_type]
        if not subset:
            continue

        for e in subset:
            e.resolve_consensus()

        labels_a = [e.annotator_1 for e in subset]
        labels_b = [e.annotator_2 for e in subset]
        kappa = cohen_kappa(labels_a, labels_b)
        n_agreed = sum(1 for e in subset if not e.needs_adjudication)
        n_disputed = len(subset) - n_agreed

        report = AgreementReport(
            event_type=event_type,
            n_events=len(subset),
            n_agreed=n_agreed,
            n_disputed=n_disputed,
            cohen_kappa=kappa,
            meets_threshold=kappa >= KAPPA_THRESHOLD,
        )
        reports[event_type] = report

        if not report.meets_threshold:
            logger.warning(
                f"Held-out event agreement for {event_type} below threshold: "
                f"κ={kappa:.3f} < {KAPPA_THRESHOLD}. "
                f"Consensus labels for this type are unreliable until adjudicated."
            )

    return reports


def consensus_ground_truth(
    events: list[HeldOutEvent],
    event_type: str,
) -> dict[str, int]:
    """Return {thread_id: consensus_label} for agreed events of a type.

    Disputed events (needs_adjudication) are excluded.
    """
    gt: dict[str, int] = {}
    for e in events:
        if e.event_type != event_type:
            continue
        if e.needs_adjudication or e.consensus_label is None:
            continue
        gt[e.thread_id] = e.consensus_label
    return gt


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def save_events(events: list[HeldOutEvent], path: str | Path) -> Path:
    """Save held-out events to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(asdict(e)) + "\n")
    logger.info(f"Saved {len(events)} held-out events to {path}")
    return path


def load_events(path: str | Path) -> list[HeldOutEvent]:
    """Load held-out events from JSONL.

    Expected schema (one JSON object per line):
        {"event_id","thread_id","event_type",
         "annotator_1","annotator_2",
         "annotator_1_notes","annotator_2_notes",
         "consensus_label","needs_adjudication"}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out events file not found: {path}. "
            f"Run the annotation protocol (outline §5.3) or use the LLM dual "
            f"annotator to generate draft annotations."
        )
    events: list[HeldOutEvent] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            events.append(HeldOutEvent(**data))
    logger.info(f"Loaded {len(events)} held-out events from {path}")
    return events


# ----------------------------------------------------------------------
# LLM-based dual annotator (draft generation)
# ----------------------------------------------------------------------

# Two distinct annotator personas to encourage independent coding
_ANNOTATOR_A_PROMPT = """You are a discourse analyst (Annotator A). Code whether the event occurred in the thread below.

Event type: {event_type}
Definitions:
- conflict: at least one participant engages in oppositional/conflict behavior
- persuasion: persuasion succeeds (e.g. a delta is awarded, someone concedes)
- escalation: conflict intensity rises across the exchange (multiple conflict participants)
- consensus: participants reach agreement / common ground

Thread topic: {topic}
Thread exchange:
{exchange}

Respond with JSON: {{"label": 0 or 1, "notes": "one-sentence justification"}}
Output ONLY the JSON."""

_ANNOTATOR_B_PROMPT = """You are an independent social-media moderator (Annotator B). Independently judge whether the event occurred.

Event type: {event_type}
Coding rubric:
- conflict: oppositional/confrontational behavior present
- persuasion: a viewpoint change / concession / delta award occurs
- escalation: conflict grows in intensity over the exchange
- consensus: agreement or compromise is reached

Topic: {topic}
Exchange:
{exchange}

Respond with JSON: {{"label": 0 or 1, "notes": "one-sentence justification"}}
Output ONLY the JSON."""


class LLMDualAnnotator:
    """Generate draft annotations using two independent LLM prompts.

    Output is a DRAFT requiring human adjudication before use as ground
    truth (outline §5.3 specifies human annotators).
    """

    def __init__(self, llm_client, model_name: str = "gpt-4o"):
        self.llm = llm_client
        self.model_name = model_name

    async def annotate_thread(
        self,
        thread_id: str,
        event_type: str,
        topic: str,
        exchange: str,
    ) -> HeldOutEvent:
        """Annotate one thread/event with two independent LLM passes."""
        import re as _re

        common = {"event_type": event_type, "topic": topic, "exchange": exchange}

        resp_a = await self.llm.chat_completion(
            [{"role": "user", "content": _ANNOTATOR_A_PROMPT.format(**common)}],
            self.model_name,
            temperature=0.2,
        )
        resp_b = await self.llm.chat_completion(
            [{"role": "user", "content": _ANNOTATOR_B_PROMPT.format(**common)}],
            self.model_name,
            temperature=0.2,
        )

        def _parse(resp: str) -> tuple[int, str]:
            try:
                obj = json.loads(resp)
                return int(obj.get("label", 0)), str(obj.get("notes", ""))
            except (json.JSONDecodeError, ValueError):
                m = _re.search(r"\{.*\}", resp, _re.DOTALL)
                if m:
                    try:
                        obj = json.loads(m.group())
                        return int(obj.get("label", 0)), str(obj.get("notes", ""))
                    except (json.JSONDecodeError, ValueError):
                        pass
                return 0, "parse failure"

        label_a, notes_a = _parse(resp_a)
        label_b, notes_b = _parse(resp_b)

        event = HeldOutEvent(
            event_id=f"{event_type}_{thread_id}",
            thread_id=thread_id,
            event_type=event_type,
            annotator_1=label_a,
            annotator_2=label_b,
            annotator_1_notes=notes_a,
            annotator_2_notes=notes_b,
        )
        event.resolve_consensus()
        return event

    async def annotate_threads(
        self,
        threads: list[dict[str, Any]],
        event_types: tuple[str, ...] = ALL_EVENT_TYPES,
        max_exchange_chars: int = 2000,
    ) -> list[HeldOutEvent]:
        """Annotate many threads across all event types.

        Args:
            threads: List of {thread_id, topic, exchange} dicts.
        """
        events: list[HeldOutEvent] = []
        for t in threads:
            topic = str(t.get("topic", ""))
            exchange = str(t.get("exchange", ""))[:max_exchange_chars]
            for et in event_types:
                ev = await self.annotate_thread(
                    thread_id=t["thread_id"],
                    event_type=et,
                    topic=topic,
                    exchange=exchange,
                )
                events.append(ev)
        return events
