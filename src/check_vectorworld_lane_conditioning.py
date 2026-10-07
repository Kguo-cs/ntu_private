"""Bounded, paired audit of public VectorWorld lane-conditioned sampling.

Compare previous behavior, each correction separately, and both fixes together.
The previous Heun predictor drift is reproduced only inside this diagnostic. All variants
use the same scene counts, EMA weights and initial agent noise. Subset JSDs
are diagnostics, not estimates of the full 50k paper metrics.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse, json, pickle, sys, time
import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.smart.vectorworld.decoder import VectorWorldInitDecoder
from src.smart.scenario_dreamer.map_categories import load_category_keys, classify_category
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.vectorworld.core.utils.data_helpers import unnormalize_latents
from src.smart.scenario_dreamer.core.data_helpers import unnormalize_scene
from src.smart.metrics.generated_map import split_generated_maps
from src.smart.metrics.metric_core import DistributionAccumulator, finalize_metrics
from src.smart.metrics.official_backend import load_official_backend

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'src/waymo_data/scenario_dreamer_ae_preprocess_waymo/test'
INDEX = ROOT/'src/waymo_data/vectorworld/metadata/nocturne_compatible_keys.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes-per-group', type=int, default=8,
                        help='Scenes from each of three provenance groups; default total is 24.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--legacy-category-index', type=Path,
                        default=INDEX.with_name('nocturne_compatible_keys_native_legacy.json'),
                        help='Explicit pre-fix train+val index for paired comparison only.')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/vectorworld_lane_conditioning_audit.json')
    args = parser.parse_args()
    if args.scenes_per_group < 1 or args.batch_size < 1:
        parser.error('scenes-per-group and batch-size must be positive')
    OUTPUT = args.output
    torch.set_num_threads(2)
    free, _ = torch.cuda.mem_get_info()
    if free < 6*2**30:
        raise SystemExit('Less than 6 GiB GPU memory free; postpone diagnostic.')
    with (ROOT/'src/waymo_data/waymo_eval_set.pkl').open('rb') as f:
        manifest = pickle.load(f)
        names = [Path(str(x)).name for x in (manifest['files'] if isinstance(manifest, dict) else manifest)]
    keys = load_category_keys(INDEX)
    legacy_keys = load_category_keys(args.legacy_category_index)
    if keys.policy != 'split_aware_nocturne_whitelist':
        parser.error('Default category index must be regenerated with the split-aware policy')
    if legacy_keys.policy != 'native_vae_train_plus_val_whitelist':
        parser.error('legacy-category-index must explicitly use the pre-fix native policy')
    groups = {'ordinary_testing': [], 'native_positive_testing': [], 'heldout_validation': []}
    for name in names:
        group = 'heldout_validation' if name.startswith('validation.') else (
            'native_positive_testing' if classify_category(name, legacy_keys) else 'ordinary_testing')
        groups[group].append(name)
    # Deliberately stratified, small diagnosis; it is not a representative full-set estimate.
    if args.scenes_per_group > min(map(len, groups.values())):
        parser.error('scenes-per-group exceeds the available provenance group size')
    selected = [groups[group][i] for i in range(args.scenes_per_group) for group in groups]
    source_counts = {group: len(values) for group, values in groups.items()}
    p = SimpleNamespace(scenario_dreamer_init=True, pred_init=True, training=False)
    p._make_ego_mask = TokenProcessor._make_ego_mask
    p._get_agent_tokens = lambda types: (torch.zeros(len(types), 2, device=types.device),
        torch.zeros(len(types), 1, 3, 4, 2, device=types.device),
        torch.zeros(len(types), 1, 4, 2, device=types.device))
    p._attach_token_libraries = lambda agent: None
    print('Loading public Flow EMA and embedded VAE', flush=True)
    decoder = VectorWorldInitDecoder(p, ae_checkpoint=None,
        ldm_checkpoint=ROOT/'src/waymo_data/vectorworld/checkpoints/flow.ckpt',
        training_mode='lane_conditioned', generation_mode='lane_conditioned',
        map_category_index=INDEX, use_ema=True).eval()
    updates = decoder.ema.num_updates
    for param, shadow in zip(decoder.diff_model.parameters(), decoder.ema.shadow_params):
        param.data = shadow
    # Install EMA once, avoiding a second GPU shadow/restore copy.
    decoder.ema = None
    decoder.to('cuda')
    model = decoder.diff_model
    field = model._cfg_vector_field
    s = decoder.cfg.dataset
    physical_keys = ('fov', 'min_speed', 'max_speed', 'min_length', 'max_length',
        'min_width', 'max_width', 'min_lane_x', 'max_lane_x', 'min_lane_y', 'max_lane_y')
    variants = {
        'previous': dict(legacy_heun=True, labels='native', precision='medium'),
        'fixed_labels_only': dict(legacy_heun=True, labels='split_aware', precision='medium'),
        'fixed_heun_only': dict(legacy_heun=False, labels='native', precision='medium'),
        'fixed': dict(legacy_heun=False, labels='split_aware', precision='medium')}
    if model.flow_solver != 'heun':
        raise ValueError('This paired diagnostic requires the published Heun solver')
    acc = {k: {m: DistributionAccumulator() for m in ('reference', 'reconstructed')} for k in variants}
    gt = DistributionAccumulator()
    traces = {k: [] for k in variants}
    diffs = {k: {x: [] for x in ('position_m', 'heading_rad', 'speed_mps', 'length_m', 'width_m')}
        for k in variants if k != 'previous'}
    type_changes = Counter()
    label_counts = {k: Counter() for k in variants}
    backend = load_official_backend()
    start = time.monotonic()

    @torch.inference_mode()
    def sample(data, variant, seed):
        settings = variants[variant]
        torch.set_float32_matmul_precision(settings['precision'])
        trace = []
        fixed = data['lane'].latents[:, None]
        predictor = {}
        def traced_field(a, l, d, ta, tl):
            received = l
            corrector = len(trace) % 2 == 1
            if settings['legacy_heun'] and corrector:
                # Reconstruct the old lane predictor; the production sampler
                # now correctly clamps it before calling this field.
                dt = predictor['time'][0] - ta[0]
                l = fixed - dt * predictor['velocity']
            trace.append({'call': len(trace), 't': float(ta[0]),
                'received_lane_max_abs': float((received-fixed).abs().max()),
                'used_lane_max_abs': float((l-fixed).abs().max()),
                'used_lane_rms': float((l-fixed).square().mean().sqrt())})
            result = field(a, l, d, ta, tl)
            if not corrector:
                predictor['time'], predictor['velocity'] = ta, result[1]
            return result
        model._cfg_vector_field = traced_field
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            a, l = model(data, mode='lane_conditioned')
        finally:
            model._cfg_vector_field = field
        if not torch.equal(l, data['lane'].latents):
            raise AssertionError('Final lane differs from its condition')
        a, l = unnormalize_latents(a, l, s.agent_latents_mean, s.agent_latents_std,
            s.lane_latents_mean, s.lane_latents_std)
        states, lanes, types, _, edges = decoder.autoencoder.forward_decoder_with_motion(a, l, data)
        states, lanes = unnormalize_scene(states, lanes, **{key: s[key] for key in physical_keys})
        maps = split_generated_maps({'coordinate_frame':'sd_local', 'road_points':lanes,
            'road_connection_types':edges, 'edge_index_lane_to_lane':data['lane','to','lane'].edge_index,
            'batch':data['lane'].batch, 'lg_type':data.lg_type}, data.batch_size)
        traces[variant].append(trace)
        return states[:, :7].cpu(), types.cpu(), maps

    for bi, offset in enumerate(range(0, len(selected), args.batch_size)):
        batch_names = selected[offset:offset+args.batch_size]
        raw = []
        for name in batch_names:
            with (DATA/name).open('rb') as f:
                scene = pickle.load(f)
            if int(scene['lg_type']) != 0:
                raise AssertionError('Expected non-partitioned scene')
            raw.append(scene)
            gt.update(backend.convert_data_to_unified_format(scene, 'waymo'), backend)
        batch = Batch.from_data_list([HeteroData(adapt_preprocessed_scene(r,n))
            for r,n in zip(raw,batch_names)]).to('cuda')
        _, agent = TokenProcessor.process_data(p, batch)
        agent['tokenized_map'] = {}
        torch.set_float32_matmul_precision('medium')
        with torch.inference_mode():
            data, *_ = decoder._encode_lanes(agent)
        split_ids = data.map_id.clone()
        native_ids = data.map_id.new_tensor([classify_category(name, legacy_keys) for name in batch_names])
        baseline = None
        for variant, settings in variants.items():
            data.map_id = native_ids if settings['labels']=='native' else split_ids
            label_counts[variant].update(data.map_id.tolist())
            states, types, maps = sample(data, variant, args.seed+bi)
            if variant == 'previous':
                baseline = (states, types)
            else:
                before, before_types = baseline
                delta = states-before
                diffs[variant]['position_m'].extend(delta[:,:2].norm(dim=-1).tolist())
                for label, col in [('speed_mps',2), ('length_m',5), ('width_m',6)]:
                    diffs[variant][label].extend(delta[:,col].abs().tolist())
                angle = torch.atan2(states[:,4],states[:,3])-torch.atan2(before[:,4],before[:,3])
                diffs[variant]['heading_rad'].extend(torch.atan2(angle.sin(),angle.cos()).abs().tolist())
                type_changes[variant] += int((types != before_types).sum())
            for i, scene in enumerate(raw):
                mask = (data['agent'].batch == i).cpu()
                attrs = dict(agent_states=states[mask].numpy(), agent_types=np.eye(3)[types[mask].numpy()])
                for metric_map, road in [('reference', scene), ('reconstructed', maps[i])]:
                    payload = dict(road, **attrs)
                    acc[variant][metric_map].update(backend.convert_data_to_unified_format(payload, 'waymo'),
                        backend, collision=True)
        print(f'Compared {offset+len(raw)}/{len(selected)} scenes, elapsed {time.monotonic()-start:.1f}s', flush=True)
    summary = {variant: {key: {'mean':float(np.mean(values)), 'p95':float(np.percentile(values,95)),
        'max':float(np.max(values))} for key,values in values.items()} for variant,values in diffs.items()}
    report = {'scope':'Stratified small-subset diagnosis; fixed counts, EMA weights and agent noise; not full-set paper reproduction',
        'seed':args.seed, 'batch_size':args.batch_size, 'num_scenes':len(selected), 'scenes':selected, 'official_source_counts':source_counts,
        'ema_updates':updates, 'sampling_steps':model.n_steps, 'solver':model.flow_solver,
        'guidance_scale':float(decoder.cfg.train.guidance_scale), 'torch':torch.__version__,
        'variants':variants, 'current_category_policy':keys.policy, 'label_policy_note':'Default labels use original split and all three Nocturne whitelists; native labels are retained only as an explicit diagnostic comparison', 'label_counts':{k:dict(v) for k,v in label_counts.items()},
        'physical_output_difference_from_previous':summary, 'type_changes':dict(type_changes),
        'lane_field_call_traces':traces,
        'metrics_vs_same_subset_gt':{k:{m:finalize_metrics(v,gt) for m,v in metrics.items()}
            for k,metrics in acc.items()},
        'sample_statistics':{k:{m:{'vehicles':v.num_vehicles,'onroad_pct':v.totals[1]/v.num_vehicles*100,
            'collision_pct':v.num_colliding/v.num_vehicles*100} for m,v in metrics.items()}
            for k,metrics in acc.items()},
        'elapsed_seconds':time.monotonic()-start}
    for name in ('fixed_heun_only', 'fixed'):
        if any(call['used_lane_max_abs'] != 0 for batch in traces[name] for call in batch):
            raise AssertionError('Corrected sampler changed lane conditions at a field evaluation')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print('Report: '+str(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
