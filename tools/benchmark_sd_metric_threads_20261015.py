"""CPU-only exact-histogram benchmark; leaves production evaluation unchanged."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.smart.metrics.real_cache import CachedReferenceStore, accumulator_state, add_statistics
    from src.smart.metrics.metric_core import DistributionAccumulator
    store = CachedReferenceStore(ROOT / 'src/waymo_data/sd_real_metric_cache.sqlite', lru_size=0)
    scenes = [store.get(name)[1] for name in store.files[25000:25512]]
    official = store.official
    store.close()

    def calculate(block):
        acc = DistributionAccumulator()
        for scene in block:
            acc.update(scene, official, collision=True)
        return accumulator_state(acc)

    blocks = [scenes[i:i + 8] for i in range(0, len(scenes), 8)]
    observations = []
    reference = None
    for workers in (1, 2, 4):
        start = time.monotonic()
        if workers == 1:
            result = calculate(scenes)
        else:
            merged = DistributionAccumulator()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for statistics in pool.map(calculate, blocks):
                    add_statistics(merged, statistics)
            result = accumulator_state(merged)
        elapsed = time.monotonic() - start
        if reference is None:
            reference = result
        assert result == reference, 'Histogram/count mismatch; do not adopt threading'
        observations.append(dict(workers=workers, seconds=elapsed,
            exact_integer_histograms_and_counts=True, scenes=len(scenes)))
        print(json.dumps(observations[-1]), flush=True)
    base = ROOT / 'output/reports/init_diffusion_continuous_20261014'
    (base / 'sd_metric_thread_benchmark_20261015.json').write_text(json.dumps(dict(
        status='BENCHMARK_COMPLETE', workload='512 real reference scenes treated as generated',
        collision_included=True, GPU_used=False, production_code_changed=False,
        observations=observations,
        caveat='Microbenchmark only; generated-scene validation and end-to-end timing required before adoption.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
