"""Materialize native configs only from a complete verified overnight decision.

No model is instantiated. Checkpoint metadata is read in an isolated CPU
process; Hydra composes temporary configs before any real output is written.
The short search does not prove a128epoch optimum. Evaluation requires an
explicit saved checkpoint through the environment or a Hydra CLI override.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continue_overnight_init_diffusion as controller
from tools.runtime_completion import verify_completion

ART = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011'
NAME = 'init_diffusion_lane_conditioned_overnight_best'
TRAIN_SEEDS = (4817, 5817)
GEN_SEEDS = (2817, 3817)
CHECKPOINT_ENV = 'INIT_DIFFUSION_OVERNIGHT_CKPT'
LEGACY_DENOISER_OPTIONS = {
    'denoiser_hidden_dim': 256, 'denoiser_num_layers': 3,
    'state_feature_normalization': 'none', 'speed_x0_parameterization': 'identity',
    'speed_head_kind': 'shared', 'speed_head_detach_context': False,
    'map_condition_types': 'road_edges',
}
NEW_COMPATIBLE_OPTIONS = dict(LEGACY_DENOISER_OPTIONS, matching_weights=None,
                            time_sample_power=1., distribution_loss=None, sampling_steps=20)
EGO_FIELDS = ('position', 'heading', 'shape', 'velocity', 'type')
PHYSICAL_DATA_FIELDS = ('train_raw_dir', 'val_raw_dir', 'test_raw_dir', 'val_tfrecords_splitted',
                        'scenario_dreamer_preprocessed', 'scenario_dreamer_train_non_partitioned_only',
                        'scenario_dreamer_train_latent_cache', 'scenario_dreamer_val_latent_cache',
                        'scenario_dreamer_test_latent_cache')
CHECKPOINT_PROVENANCE = ('seed', 'batch_size', 'base_train_offset', 'map_frozen', 'calibration_sha256',
                        'map_checkpoint_sha256', 'initial_generator_sha256', 'initial_map_sha256', 'train_order_sha256')
SIDECAR_PROVENANCE = CHECKPOINT_PROVENANCE + ('actual_train_offset', 'initialization_scope', 'resumed_from',
                         'resume_global_step', 'optimizer_continued', 'train_manifest_sha256', 'eval_manifest_sha256',
                         'lr', 'gradient_clip_val')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def evidence(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path)}


def plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def portable(value):
    if isinstance(value, str) and value.startswith(str(ROOT) + '/'):
        return '${sim_root:}/' + str(Path(value).relative_to(ROOT))
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    return value


def assessment(rows, controls):
    result = controller.compare(controller.aggregate(rows), controller.aggregate(controls))
    per_seed = {str(seed): controller.compare(controller.aggregate([r for r in rows if r['seed'] == seed]),
                     controller.aggregate([r for r in controls if r['seed'] == seed])) for seed in TRAIN_SEEDS}
    result.update(per_training_seed=per_seed, aggregate_gates_pass=result['eligible'],
                  each_training_seed_gates_pass=all(v['eligible'] for v in per_seed.values()))
    result['eligible'] &= result['each_training_seed_gates_pass']
    return result


def validate(decision, protocol, planned_arms):
    """Recompute selection from all eight actual rows, not declared rankings."""
    require(isinstance(decision, dict) and decision.get('complete_matrix') is True and decision.get('selected'),
            'Cannot materialize a winner from incomplete overnight confirmation')
    challenger = protocol.get('challenger')
    require(protocol.get('control') == 'control' and isinstance(challenger, str) and challenger != 'control',
            'Frozen protocol must name a distinct control/challenger pair')
    require(tuple(protocol.get('train_seeds', ())) == TRAIN_SEEDS and tuple(protocol.get('generation_seeds', ())) == GEN_SEEDS,
            'Frozen TRAIN/GEN plans differ from the two-by-two confirmation matrix')
    require(decision.get('frozen_challenger') == challenger and protocol.get('full50000_no_reselection') is True,
            'Decision does not retain the frozen challenger/no-reselection protocol')
    planned = {arm['name']: arm for arm in planned_arms}
    names = ('control', challenger)
    require(len(planned) == len(planned_arms) and all(name in planned for name in names),
            'Frozen arms missing or duplicated in actual search plan')
    exposure = protocol.get('actual_final_exposure_per_TRAIN')
    require(type(exposure) is int and exposure > 0 and decision.get('actual_final_exposure_per_TRAIN') == exposure,
            'Decision exposure differs from the frozen final budget')
    rows = decision.get('rows')
    require(isinstance(rows, list) and len(rows) == 8 and all(isinstance(r, dict) for r in rows),
            'Expected both recipes x two TRAIN x two GEN actual endpoint rows')
    expected = {(name, seed, gen) for name in names for seed in TRAIN_SEEDS for gen in GEN_SEEDS}
    require({(r.get('variant'), r.get('seed'), r.get('eval_seed')) for r in rows} == expected,
            'Expected eight unique planned TRAIN/GEN cells; no missing or duplicate observations')
    arms = {name: [r for r in rows if r['variant'] == name] for name in names}
    for name, armrows in arms.items():
        batch = planned[name].get('batch')
        require(batch in (256, 512) and type(batch) is int, 'Unsupported planned batch method')
        steps = protocol.get('final_budget_B' + str(batch))
        require(type(steps) is int and steps > 0 and controller.actual_exposure(steps, batch) == exposure,
                'Planned batch/update budget does not match exact frozen exposure')
        for seed in TRAIN_SEEDS:
            group = [r for r in armrows if r['seed'] == seed]
            controller.validate_rows(group, name=name, seed=seed, steps=steps, scenes=4096,
                                     batch=batch, generations=GEN_SEEDS)
            for row in group:
                require(row.get('evaluation_only') is False and row.get('resumed_from') is None
                        and row.get('resume_global_step') == 0 and row.get('optimizer_continued') is False
                        and row.get('actual_train_offset') == 0 and row.get('base_train_offset') == 0,
                        'Final source must be fresh G1/optimizer/EMA, not continuation/evaluation-only')
                require(Path(row.get('checkpoint', '')).name in (f'step_{steps}.ckpt', 'last.ckpt'),
                        'Endpoint checkpoint filename/budget differs from the frozen plan')
                require(all(isinstance(row.get(k), str) and len(row[k]) == 64
                            and all(c in '0123456789abcdef' for c in row[k]) for k in
                            ('initial_generator_sha256', 'initial_map_sha256', 'train_order_sha256',
                             'train_manifest_sha256', 'calibration_sha256', 'map_checkpoint_sha256', 'eval_manifest_sha256')),
                        'Missing or invalid actual initialization/data fingerprints')
            require(len({r['checkpoint'] for r in group}) == 1, 'GEN repeats must use the same trained checkpoint')
            require(len({r['initial_generator_sha256'] for r in group}) == 1,
                    'GEN repeats have different fresh G1 identities')
            for key in ('initial_map_sha256', 'train_order_sha256', 'train_manifest_sha256',
                        'calibration_sha256', 'map_checkpoint_sha256', 'eval_manifest_sha256'):
                require(len({r[key] for r in group}) == 1, f'GEN repeats have different provenance: {key}')
        require(len({r['initial_generator_sha256'] for r in armrows}) == 2,
                'Confirmation TRAIN seeds did not produce distinct fresh G1 identities')
    for seed in TRAIN_SEEDS:
        pair = [r for r in rows if r['seed'] == seed]
        for key in ('initial_map_sha256', 'train_order_sha256', 'train_manifest_sha256', 'eval_manifest_sha256'):
            require(len({r[key] for r in pair}) == 1, f'Paired TRAIN arms have different data/map provenance: {key}')
    comparisons = {name: assessment(arms[name], arms['control']) for name in names}
    improvements = [(v['R'], v['R_attr'], name) for name, v in comparisons.items() if v['eligible'] and v['R'] < 1]
    selected = min(improvements)[2] if improvements else 'control'
    require(comparisons[selected]['eligible'], 'Even the fallback control fails a pooled or per-TRAIN gate')
    require(decision['selected'] == selected and decision.get('selected_arm') == planned[selected],
            'Declared selection/selected-arm config disagrees with the frozen primary decision and planned arm')
    return next(r for r in arms[selected] if r['seed'] == TRAIN_SEEDS[0] and r['eval_seed'] == GEN_SEEDS[0])


def validate_source_config(source, arm, row, base=None):
    """Check actual full model config against the stage base and planned arm."""
    cfg = plain(source)
    model = cfg['model']['model_config']
    init = model['decoder']['init_diffusion']
    require(cfg.get('action') == 'fit' and cfg.get('ckpt_path') is None and cfg.get('seed') == row['seed'],
            'Source config is not the fresh fitting run identified by the row')
    require(model.get('initial_map_checkpoint') == str(Path(arm['map']).resolve()) and model.get('finetune') is True,
            'Source must load only the declared fixed map through finetune=true')
    require(model['decoder'].get('init_decoder') == 'flow' and model['decoder'].get('sep_map') is False
            and model['decoder'].get('initial_scene_only') is True
            and model['decoder'].get('dis_a2a_radius', 0) <= 0,
            'Source is not the supervised shared fixed-map InitDiffusion path')
    require(all(model['token_processor'].get(k) is True for k in ('pred_init', 'learn_init', 'scenario_dreamer_init')),
            'Source token processor is not instantaneous initialization supervision')
    require(model['token_processor'].get('init_map_crop') == 'square'
            and model['token_processor'].get('init_map_half_extent') == 50., 'Source physical map crop changed')
    require(model.get('sd_use_cached_evaluator') is True and model.get('sd_reference_mode') == 'matched'
            and model.get('sd_require_full_set') is False, 'Source metric/reference condition changed')
    require(cfg['data'].get('train_batch_size') == arm['batch'] == row['batch_size'], 'Source batch differs from arm/row')
    require(cfg['trainer'].get('precision') == '32-true' and cfg['trainer'].get('accumulate_grad_batches') == 1,
            'Source training precision/gradient accumulation changed')
    require(model.get('lr_total_steps') == 128 and init.get('use_ema') is True,
            'Source is not the declared128epoch scheduler/EMA method')
    require(init.get('component_loss') == arm['recipe'], 'Source loss recipe differs from actual planned arm')
    require(init.get('normalizer_path') is None, 'Quick source must use calibrated buffers, not another startup normalizer')
    require(row.get('lr') == model.get('lr') == arm.get('lr', .0005), 'Source learning rate differs from planned arm/row')
    require(row.get('gradient_clip_val') == cfg['trainer'].get('gradient_clip_val'), 'Source clip differs from the actual row')
    for path, value in arm['config'].items():
        require(path.startswith('model.model_config.'), 'Planned override is outside model settings')
        require(OmegaConf.select(OmegaConf.create(cfg), path) == value, f'Actual config differs from planned option: {path}')
    if base is not None:
        expected = OmegaConf.create(plain(base))
        for path, value in arm['config'].items():
            OmegaConf.update(expected, path, value, force_add=True)
        expected.model.model_config.lr = arm.get('lr', .0005)
        require(plain(expected.model) == cfg['model'], 'Full actual model differs from stage base + planned overrides')
    return cfg


def validate_checkpoint_metadata(metadata, row, arm, source):
    """Pure CPU-fixture seam; actual reading is isolated from YAML/Hydra."""
    cfg = plain(source)
    require(metadata.get('global_step') == row['budget'] and metadata.get('ema_num_updates') == row['budget']
            and metadata.get('ema_shadow_count', 0) > 0, 'Checkpoint/EMA budget is not the actual endpoint')
    require(metadata.get('optimizer_step_min') == row['budget'] == metadata.get('optimizer_step_max'),
            'Checkpoint optimizer steps differ from the endpoint budget')
    saved = metadata.get('scratch_sweep', {})
    # ScratchDiagnostics deliberately stores a smaller constructor metadata
    # subset; fresh-run scope and membership are verified in its hashed sidecar.
    for key in CHECKPOINT_PROVENANCE:
        require(key in row and saved.get(key) == row[key], f'Checkpoint provenance differs from row: {key}')
    for key in set(SIDECAR_PROVENANCE) - set(CHECKPOINT_PROVENANCE):
        if key in saved:
            require(saved[key] == row[key], f'Checkpoint optional provenance differs from row: {key}')
    require(saved.get('saved_global_step') == row['budget'], 'Checkpoint saved_global_step differs from row')
    require(metadata.get('map_tensor_sha256') == row['initial_map_sha256'], 'Fixed map weights changed during source training')
    require(metadata.get('normalizer_matches') is True and metadata.get('cuda_initialized') is False
            and metadata.get('shared_map_ema_absent') is True, 'CPU checkpoint normalizer/map inspection failed')
    init = cfg['model']['model_config']['decoder']['init_diffusion']
    extra = metadata.get('init_extra', {})
    require(extra.get('component_loss') == init['component_loss'], 'Checkpoint loss differs from actual config')
    options = {k: init.get(k) if init.get(k) is not None else v for k, v in LEGACY_DENOISER_OPTIONS.items()}
    require(dict(LEGACY_DENOISER_OPTIONS, **extra.get('denoiser_options', {})) == options,
            'Checkpoint denoiser options differ from the actual source config')
    require(extra.get('size_representation', 'linear') == init.get('size_representation', 'linear')
            and extra.get('ego_context_heading_encoding', 'angle') == init.get('ego_context_heading_encoding', 'angle'),
            'Checkpoint size/context representation differs from actual config')
    for key, default in (('generate_type', False), ('type_process', 'separate'), ('type_match_weight', 1.),
                         ('type_noise_stats', 'standard'), ('type_noise_std_floor', .01), ('use_ego_embedding', False)):
        require(extra.get(key, default) == init.get(key, default), f'Checkpoint type/ego option differs: {key}')
    conditions = {field: init.get('fix_ego_' + field) if init.get('fix_ego_' + field) is not None
                  else init.get('fix_ego', True) for field in EGO_FIELDS}
    require(extra.get('ego_conditioning') == conditions, 'Checkpoint per-field ego conditions differ from source')
    return metadata


def validate_initialization_sidecar(initialization, row):
    for key in SIDECAR_PROVENANCE:
        require(key in initialization and initialization[key] == row[key], f'Initialization sidecar differs from row: {key}')
    return initialization


def validate_execution_plan(execution, arm, row):
    require(execution.get('arm') == arm and execution.get('seed') == row['seed'],
            'Actual executed arm/seed differs from the search plan')
    command = execution.get('command')
    require(isinstance(command, list) and '--allow-multiple-epochs' in command
            and not any(flag in command for flag in ('--evaluation-checkpoints', '--resume-checkpoints', '--resume-map')),
            'Actual command was not a fresh multi-epoch fitting invocation')
    flags = {'--map-checkpoint': arm['map'], '--normalizer-json': arm['normalizer'],
             '--batch-size': str(arm['batch']), '--scenes': '4096', '--seed': str(row['seed']),
             '--steps': str(row['budget']), '--variants': arm['name'], '--evaluation-weights': 'ema',
             '--eval-seed': str(GEN_SEEDS[0]), '--extra-eval-seed': str(GEN_SEEDS[1])}
    for key, value in flags.items():
        require(command.count(key) == 1 and command.index(key) + 1 < len(command)
                and command[command.index(key) + 1] == value, f'Actual command differs from plan: {key}')
    expected = {'--recipe': arm['recipe'], '--variant-file': {arm['name']: {
                '__config': arm['config'], '__lr': arm.get('lr', .0005)}}}
    recorded = execution.get('fingerprints', {}).get('design_inputs', {})
    for flag, value in expected.items():
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'Missing actual design command flag')
        path = command[command.index(flag) + 1]
        require(read(path) == value and recorded.get(path) == controller.digest(value),
                'Actual design input differs from the planned recipe/options')
    return execution


def inspect_checkpoint(path, normalizer, python=sys.executable):
    """Trusted local checkpoint, pure torch child with CPU map_location."""
    script = '''import gc,hashlib,json,sys,torch
gc.disable()
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True)
s=c["state_dict"];e=s.get("encoder.init_decoder._extra_state",{});ema=e.get("ema") or {}
d=hashlib.sha256()
for k,t in sorted(s.items()):
 if k.startswith("encoder.map_encoder.") and torch.is_tensor(t):
  d.update(k.encode());d.update(t.contiguous().numpy().tobytes())
n=json.load(open(sys.argv[2]));ok=True
for k in ("normal_mean","normal_scale"):
 t=s["encoder.init_decoder.G1.model."+k]
 ok=ok and torch.equal(t,torch.as_tensor(n[k],dtype=t.dtype).reshape(1,-1))
steps=[int(v["step"]) for o in c.get("optimizer_states",[]) for v in o["state"].values() if "step" in v]
out={"global_step":c.get("global_step"),"scratch_sweep":c.get("scratch_sweep",{}),
"ema_num_updates":ema.get("num_updates"),"ema_shadow_count":len(ema.get("shadow_params",[])),
"optimizer_step_min":min(steps) if steps else None,"optimizer_step_max":max(steps) if steps else None,
"map_tensor_sha256":d.hexdigest(),"normalizer_matches":bool(ok),"cuda_initialized":torch.cuda.is_initialized(),
"shared_map_ema_absent":s.get("encoder._extra_state",{}).get("initial_map_ema") is None,
"init_extra":{k:v for k,v in e.items() if k not in ("ema","ema_parameter_names")}}
print(json.dumps(out,allow_nan=False))'''
    result = subprocess.run([str(python), '-X', 'faulthandler', '-c', script, str(path), str(normalizer)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, check=False)
    require(result.returncode == 0, 'Isolated CPU checkpoint inspection failed: ' + result.stderr[-1500:])
    return json.loads(result.stdout)


def audit_runtime_sources(execution, folder):
    """Keep process-runtime differences visible while verifying every wrapper."""
    protocol = read(Path(folder) / 'protocol.json')
    output = {'cyclic_gc_policy': protocol.get('cyclic_gc_policy'),
              'young_gc_interval': protocol.get('young_gc_interval'),
              'runtime_cohort_is_separate_from_model_data_metric_cohort': True}
    metadata = execution.get('runtime_recovery')
    if not metadata:
        return output
    ledger = ART / 'runtime_recovery/protocol.json'
    require(Path(metadata['ledger']).resolve() == ledger.resolve(), 'Runtime stage refers to an unexpected ledger')
    recovery = read(ledger)
    require(recovery['source_freeze_sha256'] == file_hash(ART / 'source_freeze.json')
            and recovery['frozen_long_protocol_sha256'] == file_hash(ART / 'frozen_long_protocol.json'),
            'Runtime recovery altered the frozen source/budget protocol')
    frozen = read(ART / 'source_freeze.json')['source_hashes']
    require(execution['source_hashes'] == recovery['source_hashes'] == frozen,
            'Runtime stage source SHA differs from frozen objective/model/data/runner')
    require(all(file_hash(ROOT / path) == digest for path, digest in frozen.items()),
            'A frozen source changed after the recovered evaluation')
    require(metadata['wrapper_sha256'] == recovery['wrapper_sha256'] == file_hash(recovery['wrapper']),
            'Original recovery controller SHA changed')
    runtime = recovery.get('runtime_loader_source_hashes', {})
    require(runtime and metadata.get('runtime_loader_source_hashes') == runtime
            and all(file_hash(path) == digest for path, digest in runtime.items()),
            'Runtime loader entry/helper source SHA changed')
    require(execution['command'][1] == recovery['loader_wrapper'] and protocol.get('young_gc_interval') == 0
            and protocol.get('defer_cyclic_gc') is True, 'Actual entrypoint/GC policy differs from recovery ledger')
    output.update(recovery_ledger=evidence(ledger), recovery_controller=evidence(recovery['wrapper']),
                  runtime_loader_sources={path: evidence(path) for path in runtime},
                  original_source_freeze=evidence(ART / 'source_freeze.json'),
                  actual_runtime_policy=evidence(Path(folder) / 'runtime_loader_policy.json'))
    bridge_path = ART / 'runtime_recovery/verification_bridge.json'
    if bridge_path.exists():
        bridge = read(bridge_path)
        require(bridge['original_recovery_protocol_sha256'] == file_hash(ledger),
                'Verification bridge targets a different recovery protocol')
        require(bridge['source_hashes'] and all(file_hash(path) == digest for path, digest in bridge['source_hashes'].items()),
                'Completion verification bridge/helper source changed')
        output.update(verification_bridge=evidence(bridge_path),
                      verification_bridge_sources={path: evidence(path) for path in bridge['source_hashes']})
        certification = execution.get('completion_verification_bridge')
        if certification:
            require(certification['source_hashes'] == bridge['source_hashes']
                    and certification['actual_returncode'] == 0
                    and certification['actual_complete_rows'] == len(execution['completion_spec']['budgets']) * len(GEN_SEEDS),
                    'Completed-stage certification differs from actual bridge/evaluation count')
    return output


def verify_evidence(decision, protocol, planned_arms, *, checkpoint_reader=inspect_checkpoint, verify_current_source=True):
    """Read every completed stage/report/config/checkpoint, not only the winner."""
    selected = validate(decision, protocol, planned_arms)
    planned = {arm['name']: arm for arm in planned_arms}
    configs, audits, cohorts = {}, [], []
    for name in ('control', protocol['challenger']):
        arm = planned[name]
        require(Path(arm['map']).is_file() and Path(arm['normalizer']).is_file(), 'Declared map/normalizer is missing')
        map_sha, norm_sha = file_hash(arm['map']), file_hash(arm['normalizer'])
        for seed in TRAIN_SEEDS:
            rows = [r for r in decision['rows'] if r['variant'] == name and r['seed'] == seed]
            path = Path(rows[0]['checkpoint']).resolve()
            require(path.is_file(), 'Actual evidence endpoint checkpoint is missing')
            folder = path.parent.parent
            execution = read(folder / 'execution.json')
            require(execution.get('status') == 'success' and execution.get('returncode') == 0,
                    'Only a verified completed stage can produce native config evidence')
            validate_execution_plan(execution, arm, rows[0])
            spec = execution.get('completion_spec') or {}
            require(spec.get('name') == name and spec.get('seed') == seed and spec.get('steps') == rows[0]['budget']
                    and spec.get('batch') == arm['batch'] and spec.get('scenes') == 4096
                    and tuple(spec.get('generations', ())) == GEN_SEEDS, 'Stage planned endpoint spec differs from rows/arm')
            actual, artifacts = verify_completion(folder, folder.parent / (folder.name + '.log'), spec)
            require(artifacts == execution.get('artifacts'), 'Completed stage evidence fingerprints changed')
            endpoint = [r for r in actual if r['budget'] == spec['steps']]
            require(sorted(endpoint, key=lambda r: r['eval_seed']) == sorted(rows, key=lambda r: r['eval_seed']),
                    'Decision rows differ from actual completed summary endpoints')
            cfg = OmegaConf.load(path.parent / 'config.yaml')
            validate_source_config(cfg, arm, rows[0], OmegaConf.load(folder / 'base_config.yaml'))
            validate_initialization_sidecar(read(path.parent / 'initialization.json'), rows[0])
            runtime_audit = audit_runtime_sources(execution, folder)
            for manifest, key in ((folder / 'train_order.txt', 'train_order_sha256'),
                       (Path(cfg.data.scenario_dreamer_train_sample_list).with_suffix('.txt'), 'train_manifest_sha256'),
                       (Path(cfg.data.scenario_dreamer_eval_set).with_suffix('.txt'), 'eval_manifest_sha256')):
                require(manifest.is_file() and file_hash(manifest) == rows[0][key],
                        f'Actual source input manifest differs from row: {key}')
            configs[(name, seed)] = cfg
            for row in rows:
                require(row.get('map_checkpoint_sha256') == map_sha and row.get('calibration_sha256') == norm_sha,
                        'Declared map/normalizer file differs from actual row')
                cohorts.append(controller.row_cohort(row))
            model_sources = {k: v for k, v in execution.get('source_hashes', {}).items()
                             if k.startswith('src/smart/') or k == 'configs/model/smart.yaml'}
            require(model_sources, 'Missing actual model/data source fingerprints')
            if verify_current_source:
                require(all((ROOT / k).is_file() and file_hash(ROOT / k) == v for k, v in model_sources.items()),
                        'Model/data source changed since completed evaluation; establish compatibility before materialization')
            metadata = checkpoint_reader(path, Path(arm['normalizer']))
            validate_checkpoint_metadata(metadata, rows[0], arm, cfg)
            checkpoint_evidence = evidence(path)
            if path.name == 'last.ckpt':
                endpoint_path = path.parent / f'step_{rows[0]["budget"]}.ckpt'
                require(endpoint_path.is_file() and file_hash(endpoint_path) == checkpoint_evidence['sha256'],
                        'Actual last.ckpt is not the saved final post-update step checkpoint')
            audits.append({'variant': name, 'TRAIN_seed': seed, 'checkpoint': checkpoint_evidence,
                           'source_config': evidence(path.parent / 'config.yaml'), 'execution': evidence(folder / 'execution.json'),
                           'source_initialization': evidence(path.parent / 'initialization.json'),
                           'raw_log_and_completion_artifacts': artifacts, 'runtime_provenance': runtime_audit,
                           'checkpoint_metadata': metadata})
    require(len({controller.digest(c) for c in cohorts}) == 1, 'Final physical input/metric/source cohorts differ')
    for name in ('control', protocol['challenger']):
        require(plain(configs[(name, TRAIN_SEEDS[0])].model) == plain(configs[(name, TRAIN_SEEDS[1])].model),
                'Same recipe model differs across confirmation TRAIN seeds')
    return selected, configs[(decision['selected'], TRAIN_SEEDS[0])], audits


def build_configs(source, arm, *, eval_checkpoint=None):
    """Pure builder. Search checkpoint is recipe evidence, never fit init."""
    source = OmegaConf.create(plain(source))
    model = portable(plain(source.model))
    # Explicit constructor defaults prevent later Hydra defaults from enabling
    # features that did not exist in a legacy evidence config.
    init = model['model_config']['decoder']['init_diffusion']
    for key, value in NEW_COMPATIBLE_OPTIONS.items():
        if key not in init or (key in ('denoiser_hidden_dim', 'denoiser_num_layers', 'sampling_steps') and init[key] is None):
            init[key] = value
    model['model_config']['initial_map_checkpoint'] = portable(str(Path(arm['map']).resolve()))
    model['model_config']['finetune'] = True
    model['model_config']['lr_total_steps'] = 128
    model['model_config']['sd_reference_mode'] = 'matched'
    model['model_config']['sd_require_full_set'] = False
    model['model_config']['decoder']['init_diffusion']['normalizer_path'] = portable(str(Path(arm['normalizer']).resolve()))
    training = {'defaults': ['init_diffusion_lane_conditioned_frozen_map', '_self_'],
                'task_name': NAME, 'action': 'fit', 'ckpt_path': None, 'seed': TRAIN_SEEDS[0],
                'float32_matmul_precision': source.float32_matmul_precision,
                'model': model, 'data': {'train_batch_size': arm['batch'],
                    **{key: portable(plain(source.data[key])) for key in PHYSICAL_DATA_FIELDS if key in source.data},
                    'scenario_dreamer_train_sample_list': None, 'scenario_dreamer_eval_set': None},
                'trainer': {'max_epochs': 128, 'max_steps': -1, 'precision': source.trainer.precision,
                            'accumulate_grad_batches': source.trainer.accumulate_grad_batches,
                            'gradient_clip_val': float(source.trainer.gradient_clip_val)}}
    evaluation = {'defaults': [NAME, '_self_'], 'task_name': NAME + '_eval', 'action': 'test',
                  'ckpt_path': portable(str(Path(eval_checkpoint).resolve())) if eval_checkpoint else '${oc.env:' + CHECKPOINT_ENV + '}',
                  'seed': GEN_SEEDS[0],
                  'model': {'model_config': {'initial_map_checkpoint': None,
                            'sd_reference_mode': 'matched', 'sd_require_full_set': True, 'sd_expected_scenes': 50000,
                            'decoder': {'init_diffusion': {'normalizer_path': None}}}},
                  'data': {'scenario_dreamer_eval_set': None},
                  'trainer': {'limit_test_batches': 1., 'logger': False}, 'callbacks': None, 'logger': False}
    return training, evaluation


def config_text(cfg, *, evaluation=False):
    lines = ['# @package _global_', '# Selected by an actual two-TRAIN / two-GEN overnight comparison.',
             '# TRAIN 4817 participated in selection; TRAIN 5817 confirmed the frozen recipe.',
             '# The complete 128-epoch optimum has not been established.']
    lines += (['# Pass ckpt_path=/path/to/checkpoint.ckpt or ' + CHECKPOINT_ENV + '.',
               '# Checkpoint restores map/G1/normalizer/EMA; no startup map or normalizer load.'] if evaluation else
              ['# Native fit starts fresh G1/optimizer/EMA and initializes only the fixed AIRL map.',
               '# Native inherited shuffle=True differs from the quick fixed-order search.'])
    return '\n'.join(lines) + '\n' + yaml.safe_dump(cfg, sort_keys=False)


def compose_configs(training, evaluation, *, root=ROOT):
    """Hydra compose temp copies; never instantiate model or write best files."""
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    with tempfile.TemporaryDirectory(prefix='overnight-config-compose-') as temporary:
        cfgdir = Path(temporary) / 'configs'
        shutil.copytree(Path(root) / 'configs', cfgdir)
        for name, cfg in ((NAME, training), (NAME + '_eval', evaluation)):
            (cfgdir / 'experiment' / (name + '.yaml')).write_text(config_text(cfg, evaluation=name.endswith('_eval')))
        with initialize_config_dir(config_dir=str(cfgdir), version_base=None):
            overrides = ['paths.output_dir=' + temporary, 'paths.log_dir=' + temporary]
            fit = compose(config_name='run', overrides=['experiment=' + NAME, *overrides])
            test = compose(config_name='run', overrides=['experiment=' + NAME + '_eval', *overrides,
                            'ckpt_path=/explicit/deployed_checkpoint.ckpt'])
        fit, test = plain(fit), plain(test)
    require(fit['action'] == 'fit' and fit['ckpt_path'] is None and fit['trainer']['max_epochs'] == 128
            and fit['trainer']['max_steps'] == -1, 'Composed fit would resume or use a short-step budget')
    require(fit['model']['model_config']['finetune'] is True
            and fit['model']['model_config']['initial_map_checkpoint'], 'Composed fit lost map-only initialization')
    require(test['action'] == 'test' and test['ckpt_path'] == '/explicit/deployed_checkpoint.ckpt'
            and test['model']['model_config']['initial_map_checkpoint'] is None
            and test['model']['model_config']['decoder']['init_diffusion']['normalizer_path'] is None,
            'Composed evaluation would overwrite saved map/normalizer or omit explicit checkpoint')
    left = copy.deepcopy(fit['model']); right = copy.deepcopy(test['model'])
    for item in (left, right):
        for key in ('initial_map_checkpoint', 'sd_require_full_set', 'sd_expected_scenes'):
            item['model_config'].pop(key, None)
        item['model_config']['decoder']['init_diffusion'].pop('normalizer_path', None)
    require(left == right, 'Evaluation model/conditioning differs from selected training config')
    expected = plain(OmegaConf.create(training['model']))
    for key, value in expected['model_config'].items():
        require(fit['model']['model_config'][key] == value, f'Composition changed selected model option: {key}')
    return fit, test


def write_outputs(outputs, *, overwrite=False):
    for path, text in outputs.items():
        if path.exists() and path.read_text() != text and not overwrite:
            raise FileExistsError(f'Refusing to replace existing content without --overwrite: {path}')
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', delete=False) as handle:
            handle.write(text); name = handle.name
        try:
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=Path, default=ART / 'final_selection.json')
    parser.add_argument('--frozen-protocol', type=Path, default=ART / 'frozen_long_protocol.json')
    parser.add_argument('--search-protocol', type=Path, default=ART / 'search_protocol.json')
    parser.add_argument('--eval-checkpoint', type=Path, help='Explicit deployed checkpoint; otherwise require env/ckpt_path CLI')
    parser.add_argument('--checkpoint-python', default=sys.executable, help='sim torch Python for isolated CPU metadata reads')
    parser.add_argument('--dry-run', action='store_true', help='Verify/compose completely; write no real configs')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        decision, protocol, search = read(args.decision), read(args.frozen_protocol), read(args.search_protocol)
        row, source, audits = verify_evidence(decision, protocol, search['arms'],
            checkpoint_reader=lambda p, n: inspect_checkpoint(p, n, python=args.checkpoint_python))
        if args.eval_checkpoint is not None:
            require(args.eval_checkpoint.is_file(), 'Explicit evaluation checkpoint is missing')
        training, evaluation = build_configs(source, decision['selected_arm'], eval_checkpoint=args.eval_checkpoint)
        fit, test = compose_configs(training, evaluation)
        artifact = {'selected': decision['selected'], 'evidence_checkpoint': row['checkpoint'],
                    'fit_config': str(ROOT / 'configs/experiment' / (NAME + '.yaml')),
                    'eval_config': str(ROOT / 'configs/experiment' / (NAME + '_eval.yaml')),
                    'fit_resumes_evidence_checkpoint': False, 'initialization': 'only fixed map; fresh G1/optimizer/EMA',
                    'full128epoch_optimum_proven': False, 'independent_confirmations': 1,
                    'selection_reused_TRAIN_seed': 4817, 'confirmation_TRAIN_seed': 5817,
                    'criterion': 'Equal-TRAIN arm mean ratios, GEN averaged first; pooled and each TRAIN gates required',
                    'decision': evidence(args.decision), 'frozen_protocol': evidence(args.frozen_protocol),
                    'search_protocol': evidence(args.search_protocol), 'source_audits': audits,
                    'Hydra_composition_verified': True,
                    'native_shuffle': fit['data']['shuffle'], 'quick_search_shuffle': plain(source)['data']['shuffle'],
                    'evaluation_requires_explicit_checkpoint': args.eval_checkpoint is None,
                    'evaluation_CLI': 'python src/run.py experiment=' + NAME + '_eval ckpt_path=/path/to/checkpoint.ckpt',
                    'evaluation_env': CHECKPOINT_ENV + '=/path/to/checkpoint.ckpt python src/run.py experiment=' + NAME + '_eval'}
        if not args.dry_run:
            outputs = {ROOT / 'configs/experiment' / (NAME + '.yaml'): config_text(training),
                       ROOT / 'configs/experiment' / (NAME + '_eval.yaml'): config_text(evaluation, evaluation=True),
                       ART / 'native_best_materialization.json': json.dumps(artifact, indent=2, allow_nan=False) + '\n'}
            write_outputs(outputs, overwrite=args.overwrite)
        print(('VERIFIED_ONLY ' if args.dry_run else 'MATERIALIZED ') + decision['selected'])
        return artifact
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
