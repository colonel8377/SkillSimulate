"""Clustering-Only Descriptive Persona baseline.

Outline §5.2 Condition 6: shares CADP's Step 1 clustering structure but
uses a descriptive persona per cluster instead of a distilled .skill file.
This isolates the contribution of "clustering structure" vs "behavioral
rule distillation".
"""

from __future__ import annotations

from src.agents.descriptive import DescriptivePersonaAgent


class ClusteringOnlyAgent(DescriptivePersonaAgent):
    """Baseline 6: Clustering-Only Descriptive Persona.

    Same cluster allocation as CADP; only the descriptive persona built
    from that cluster's behavioral statistics is injected. No .skill rules,
    no enforcement.
    """

    def __init__(self, *args, persona_description: str = "", **kwargs):
        super().__init__(*args, persona_description=persona_description, **kwargs)
        # Explicit marker for logging / analysis
        self.condition = "clustering_only"
