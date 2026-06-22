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

    # Shared embedder: used by clustering, Tier 1, Tier 2, Tier 3, Expression DNA.
    # Any SentenceTransformer-compatible model name works.
    embedder_model: str = "BAAI/bge-large-en-v1.5"
    embedder_device: str = "auto"      # auto | cpu | cuda:N
    embedder_gpu_id: int = 0           # used when device=auto
    embedder_batch_size: int = 64

    # SIP (Semantic Information Preservation) uses a SEPARATE encoder so the
    # Linguistics layer's SIP sub-metric is NOT circular with the embedding
    # space used by Tier 1, Tier 3, clustering, and Expression DNA distillation
    # (outline §3.1 dissociation claim / §5.4 anti-leakage requirement).
    # Must be a different model family from the shared embedder.
    sip_model: str = "paraphrase-mpnet-base-v2"
    sip_device: str = "auto"           # auto | cpu | cuda:N
    sip_gpu_id: int = 1                # used when device=auto
    sip_batch_size: int = 64

    model_config = {"env_prefix": "CADP_", "env_file": ".env", "extra": "ignore"}


settings = Settings()


@lru_cache(maxsize=1)
def get_shared_embedder():
    """Shared embedder singleton — used by clustering, Tier 1/2/3, Expression DNA.

    Lives on the configured GPU by default; falls back to CPU locally.
    """
    from src.config.embedder import EmbedderWrapper, _resolve_device

    device = _resolve_device(settings.embedder_device, settings.embedder_gpu_id)
    return EmbedderWrapper(
        model_name=settings.embedder_model,
        device=device,
        batch_size=settings.embedder_batch_size,
    )


@lru_cache(maxsize=1)
def get_sip_embedder():
    """SIP embedder singleton — dedicated to the Linguistics SIP metric.

    Deliberately a DIFFERENT model family from ``get_shared_embedder`` so the
    SIP score is not circular with the embedding space CADP actively constrains
    text toward (outline §3.1 / §5.4).
    """
    from src.config.embedder import EmbedderWrapper, _resolve_device

    device = _resolve_device(settings.sip_device, settings.sip_gpu_id)
    return EmbedderWrapper(
        model_name=settings.sip_model,
        device=device,
        batch_size=settings.sip_batch_size,
    )
