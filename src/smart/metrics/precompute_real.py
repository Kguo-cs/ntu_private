"""Prepare real histograms/maps once, or audit a previously prepared cache."""
from __future__ import annotations
import argparse
from pathlib import Path
if __package__:
    from .real_cache import prepare_real_cache, CachedReferenceStore, DEFAULT_CACHE_NAME
else:
    from real_cache import prepare_real_cache, CachedReferenceStore, DEFAULT_CACHE_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',default='/home/ke/code/sim/src/waymo_data/scenario_dreamer_ae_preprocess_waymo/test', help='Official preprocessed Waymo test cache, NOT TFRecords')
    parser.add_argument('--eval-set',default='/home/ke/code/sim/src/waymo_data/waymo_eval_set.pkl', help='Trusted waymo_eval_set.pkl')
    parser.add_argument('--out',default='/home/ke/code/sim/src/waymo_data/sd_real_metric_cache.sqlite', help='Output SQLite file; default cache-root/'+DEFAULT_CACHE_NAME)
    parser.add_argument('--expected-scenes', type=int, default=50_000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verify-sources', choices=('none','stat','sha256'), default='sha256')
    parser.add_argument('--allow-missing-edge-index', action='store_true',
                        help='Explicitly allow source-major complete-graph fallback (not default)')
    args = parser.parse_args()
    if not args.out and not args.cache_root:
        parser.error('Provide --out or --cache-root')
    out = Path(args.out) if args.out else Path(args.cache_root)/DEFAULT_CACHE_NAME
    if not args.verify_only:
        if not args.cache_root or not args.eval_set:
            parser.error('Building requires --cache-root and --eval-set')
        prepare_real_cache(args.cache_root, args.eval_set, out,
            expected_scenes=args.expected_scenes, overwrite=args.overwrite,
            require_explicit_edges=not args.allow_missing_edge_index)
    store = CachedReferenceStore(out, eval_set=args.eval_set, expected_scenes=args.expected_scenes)
    if args.verify_sources != 'none':
        store.verify_sources(args.cache_root, mode=args.verify_sources)
    print('Real cache:',store.database)
    print('Real scenes:',len(store.files))
    print('Real vehicles:',store.full_real.num_vehicles)
    print('Feature counts:',store.full_real.totals.tolist())
    print('Source manifest SHA-256:',store.metadata['source_manifest_sha256'])
    print('Cached GT features and 100-point metric lanes are ready; no external repo is needed.')
    store.close()

if __name__ == '__main__':
    main()
