"""Global settings loaded from environment variables."""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    held_out_events_dir: Path = data_dir / "held_out_events"
    role_labels_dir: Path = data_dir / "role_labels"
    output_dir: Path = project_root / "outputs"
    skills_dir: Path = output_dir / "skills"
    simulations_dir: Path = output_dir / "simulations"
    results_dir: Path = output_dir / "results"

    # Model cache
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    # SIP (Semantic Information Preservation) uses a SEPARATE encoder so the
    # Linguistics layer's SIP sub-metric is NOT circular with the embedding
    # space used by Tier 1, Tier 3, clustering, and Expression DNA distillation
    # (outline §3.1 dissociation claim / §5.4 anti-leakage requirement).
    # all-MiniLM-L6-v2 and paraphrase-mpnet-base-v2 are different model
    # families trained on different objectives / data; using a distinct
    # family is the cleanest way to break the circularity. Override via
    # CADP_SIP_MODEL env var if desired.
    sip_model: str = "paraphrase-mpnet-base-v2"

    model_config = {"env_prefix": "CADP_", "env_file": ".env"}


settings = Settings()


@lru_cache(maxsize=1)
def get_shared_embedder():
    """Shared SentenceTransformer singleton — avoids repeated model loading.

    Used by Tier 1 (Expression DNA 2σ filter), Tier 3 (semantic anti-pattern
    trigger), clustering embeddings, and Expression DNA distillation. Do NOT
    reuse this for the Linguistics SIP metric — see ``get_sip_embedder``.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.sentence_transformer_model)


@lru_cache(maxsize=1)
def get_sip_embedder():
    """SentenceTransformer singleton dedicated to the Linguistics SIP metric.

    Deliberately a DIFFERENT model family from ``get_shared_embedder`` so the
    SIP score is not circular with the embedding space CADP actively constrains
    text toward (outline §3.1 / §5.4).
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.sip_model)
