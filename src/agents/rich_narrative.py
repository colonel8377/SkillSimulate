"""Rich Cluster Narrative — internal lever-1 ceiling / kill condition.

Reframe v1 (2026-07-08): the richest possible persona description built
from aggregate cluster statistics in narrative form. NO compiled .skill rules, NO
filter-retry. Tests whether distilled behavioral skills beat the
strongest lever-1 description.

If CADP fails the pre-registered viability metrics against this condition,
the method-led route stops. This is not a Scaling-Law reproduction.
"""

from __future__ import annotations

from src.agents.descriptive import DescriptivePersonaAgent


class RichNarrativeAgent(DescriptivePersonaAgent):
    """Lever-1 ceiling condition: maximalist narrative persona.

    Identical system-prompt wrapping as ``DescriptivePersonaAgent``; the
    only difference relative to DescriptivePersonaAgent is the depth of
    ``persona_description``. Relative to CADP, both information granularity
    and execution mechanism differ; this is an overall-package viability
    control, not a single-variable causal contrast.
    """

    def get_constraints_text(self) -> str:
        """Slightly stronger lever-1 constraint than ``DescriptivePersonaAgent``.

        Ladder must be strict monotone on the constraint axis:
        vanilla ("") → descriptive ("stay in character") → pop_aligned
        ("stay consistent") → **rich_narrative (stay in character based
        on the narrative)** → cadp ("MUST avoid..."). Inherits from
        DescriptivePersonaAgent would flatten C2 and C4 to identical
        text, breaking the ladder.
        """
        return (
            "Stay in character based on the narrative above — including "
            "the way you handle conflict, shift stance, and engage with "
            "your cohort."
        )


__all__ = ["RichNarrativeAgent"]
