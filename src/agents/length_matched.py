"""Length-Matched Control agent.

ARS review 2026-06-19, Devil's Advocate issue C-C / alternative explanation
E1: "Better fidelity may come purely from the larger prompt — more tokens of
behavioral context = better behavioral match, independent of any 'design
blueprint.'" A length-matched control (long random behavioral description
of equal token count) was missing.

This condition closes that gap. It receives a token-budget-matched
behavioral description built from a RANDOM OTHER cluster's stats: same
template, same token count, same form as Descriptive Persona, but broken
semantic correspondence to the agent's own cluster. If CADP (or Descriptive
Persona) beats this control, the win is notarizable to *content* rather
than *token mass*. If this control matches Descriptive Persona, the
"descriptive persona" effect is partly a token-budget artifact.

See `PopulationBuilder._create_agent` (condition == "length_matched_control")
for how the random other cluster is selected and the description built via
the existing `_build_descriptive_persona` template.
"""

from __future__ import annotations

from src.agents.descriptive import DescriptivePersonaAgent


class LengthMatchedControlAgent(DescriptivePersonaAgent):
    """Control 14: token-budget-matched random behavioral description.

    Same template and token budget as Descriptive Persona (Condition 2),
    but the description is built from a randomly-chosen other cluster's
    behavioral statistics. Isolates "matched token mass + matched form"
    (DA-E1 alternative explanation) from "matched behavioral content".

    No .skill rules, no enforcement — identical machinery to
    ClusteringOnlyAgent, differing only in the source cluster used to
    build the description.
    """

    def __init__(self, *args, persona_description: str = "", **kwargs):
        super().__init__(*args, persona_description=persona_description, **kwargs)
        # Explicit marker for logging / analysis (matches the convention
        # used by ClusteringOnlyAgent).
        self.condition = "length_matched_control"
