"""Benchmark shared embedder throughput across batch sizes.

Run remotely on a 4×4090 server to confirm GPU utilization and choose a
sensible batch size. Run locally with CADP_FORCE_CPU_EMBEDDER=1 to compare.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make repo root importable when running from scripts/
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np

from src.config.settings import get_shared_embedder


def benchmark(batch_size: int, n_batches: int = 10) -> tuple[float, float]:
    model = get_shared_embedder()
    texts = ["this is a test sentence for benchmarking embedding throughput"] * batch_size
    times: list[float] = []
    # Warmup
    model.encode(texts[:1], show_progress_bar=False)
    for _ in range(n_batches):
        t0 = time.perf_counter()
        model.encode(texts, show_progress_bar=False)
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times))
    std = float(np.std(times))
    return mean, std


def main() -> None:
    model = get_shared_embedder()
    print(
        f"Model: {model.model_name}, "
        f"dim: {model.get_sentence_embedding_dimension()}, "
        f"device: {model.device}, "
        f"batch_size config: {model.batch_size}"
    )
    print(f"{'batch':>8} {'mean_ms':>10} {'std_ms':>10} {'texts/sec':>12}")
    for bs in [1, 8, 16, 32, 64, 128, 256]:
        mean, std = benchmark(bs, n_batches=10)
        print(f"{bs:>8} {mean*1000:>10.2f} {std*1000:>10.2f} {bs/mean:>12.1f}")


if __name__ == "__main__":
    main()
