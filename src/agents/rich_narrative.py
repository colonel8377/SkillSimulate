"""Rich-Narrative Persona agent — lever-1 ceiling / kill condition.

Reframe v1 (2026-07-08): the richest possible persona description built
from the SAME cluster statistics as CADP, but in narrative form with
concrete episodes and example moves. NO compiled .skill rules, NO
filter-retry. Tests whether distilled behavioral skills beat the
strongest lever-1 description.

If CADP loses to this on Predictive Fidelity, the headline thesis fails.
Token-budget matched to a CADP skill file so length is not the confound.
"""

from __future__ import annotations

from src.agents.descriptive import DescriptivePersonaAgent


class RichNarrativeAgent(DescriptivePersonaAgent):
    """Lever-1 ceiling condition: maximalist narrative persona.

    Identical system-prompt wrapping as ``DescriptivePersonaAgent``; the
    only difference is the depth of ``persona_description`` (long
    multi-paragraph narrative vs terse bullet stats). Same data source,
    same builder path → isolates "rules vs description" as the sole
    variable in the CADP-vs-rich_narrative kill comparison.
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
