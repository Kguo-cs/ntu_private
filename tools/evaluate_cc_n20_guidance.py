"""Isolated, paired N20 guidance evaluation; planning imports no ML framework.

Run one registered cell per native TEST process. Training/checkpoints are never
modified. Each result retains its native report, histogram and observed NFE.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
ART = ROOT / 'output/reports/wandb_cc_20261013/guidance'
RUNS = ROOT / 'logs/wandb_cc_guidance_20261013'
OPTIONS = {
    'off': {'scale': 0., 't_max': 0., 'iters': 1, 'margin': 0., 'vehicles_only': True},
    'light': {'scale': .25, 't_max': .2, 'iters': 5, 'margin': 0., 'vehicles_only': True},
    'cc': {'scale': .5, 't_max': .3, 'iters': 5, 'margin': .1, 'vehicles_only': True},
}
SOURCES = ('src/smart/diffusion/scale_flow.py', 'src/smart/diffusion/denoiser.py',
           'src/smart/diffusion/initial_diffusion.py', 'src/smart/model/smart_gail.py',
           'src/smart/metrics/metric_core.py', 'tools/evaluate_cc_n20_guidance.py',
           'tools/evaluate_init_diffusion_sampler.py', 'tools/evaluate_init_diffusion_native_sampler.py')

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def make_hook(options, **kwargs):
    import lightning as L
    from tools.evaluate_init_diffusion_sampler import InferenceSamplerHookCore
    class Hook(InferenceSamplerHookCore, L.Callback):
        def on_test_start(self, trainer, model):
            super().on_test_start(trainer, model)
            flow = model.encoder.init_decoder.G1
            require(self.steps == 20, 'Only N20 is authorized')
            self.previous_guidance = dict(flow.collision_guidance)
            self.previous_exact = flow.sampling_exact_x0
            flow.collision_guidance = dict(options)
            flow.sampling_exact_x0 = False
            self.record.update(guidance=dict(flow.collision_guidance), exact_x0=False,
                               observed_NFE_per_sample=[], total_denoiser_calls=0,
                               denoiser_parameters=sum(p.numel() for p in flow.model.parameters()))
            require(self.record['denoiser_parameters'] <= 8100000, 'Parameter cap exceeded')
            def forward_pre_hook(*unused):
                self.record['total_denoiser_calls'] += 1
            self.forward_handle = flow.model.register_forward_pre_hook(forward_pre_hook)
            original = flow.sample
            def sample(*args, **kw):
                before = self.record['total_denoiser_calls']
                result = original(*args, **kw)
                calls = self.record['total_denoiser_calls'] - before
                require(calls == 20, 'Actual NFE must be20')
                self.record['observed_NFE_per_sample'].append(calls)
                return result
            flow.sample = sample
            save(self.audit_path, self.record)

        def on_test_end(self, trainer, model):
            require(len(self.record['observed_NFE_per_sample']) == self.record['sample_calls'] > 0,
                    'All generated samples must have an observed NFE')
            self.forward_handle.remove()
            from src.smart.metrics.metric_core import SPECS
            evaluator = model.sd_evaluator
            save(self.audit_path.parent / 'feature_histograms.json', {
                name: {'minimum': lo, 'maximum': hi, 'bin_width': step,
                       'generated': evaluator.generated.histograms[i].tolist(),
                       'matched_gt': evaluator.real.histograms[i].tolist()}
                for i, (name, lo, hi, step, _) in enumerate(SPECS)
            })
            flow = model.encoder.init_decoder.G1
            flow.collision_guidance = self.previous_guidance
            flow.sampling_exact_x0 = self.previous_exact
            super().on_test_end(trainer, model)
    return Hook(**kwargs)

def worker(path):
    gc.disable()
    gc.collect = lambda *unused, **kw: 0
    os.environ['SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC'] = '1'
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    from omegaconf import OmegaConf
    from tools.evaluate_init_diffusion_native_sampler import build_eval_config
    from src.run import run
    from src.smart.metrics.metric_core import SPECS
    torch.set_num_threads(4)
    cell = read(path)
    for source, sha in cell['sources'].items():
        require(digest(ROOT / source) == sha, 'Registered source changed: ' + source)
    require(digest(cell['checkpoint']) == cell['checkpoint_sha256'], 'Checkpoint changed')
    require(digest(cell['source_config']) == cell['source_config_sha256'], 'Source config changed')
    folder = Path(cell['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    require(not (folder / 'sd_agent_metrics.json').exists(), 'Never overwrite an actual native result')
    manifest = folder / 'eval_files.pkl'
    with manifest.open('wb') as handle:
        pickle.dump({'files': cell['files']}, handle)
    source = OmegaConf.to_container(OmegaConf.load(cell['source_config']), resolve=True)
    cfg = build_eval_config(source, cell, folder, manifest)
    cfg['model']['model_config']['sd_r_reference'] = None
    cfg['trainer']['enable_progress_bar'] = False
    cfg['trainer']['enable_model_summary'] = False
    cfg['callbacks']['sampler_audit'].update(
        _target_='tools.evaluate_cc_n20_guidance.make_hook', options=cell['guidance'])
    cfg = OmegaConf.create(cfg)
    OmegaConf.save(cfg, folder / 'config.yaml')
    started = time.monotonic()
    os.chdir(folder)
    run(cfg)
    report, audit = read(folder / 'sd_agent_metrics.json'), read(folder / 'sampler_audit.json')
    require(report['num_samples'] == cell['scenes'] and report['reference_mode'] == 'matched', 'Actual scope mismatch')
    require(report['unverified_generation_timestep_count'] == 0, 'Unverified reference timestep')
    require(report['decoder']['sampling_steps'] == 20 and audit['finished']
            and set(audit['observed_NFE_per_sample']) == {20}, 'Observed native N20 proof incomplete')
    require(audit['EMA_num_updates'] == cell['trained_updates'] and audit['training_loss_config_unchanged'],
            'EMA/fit objective changed')
    # Native score/report is retained; raw feature counts avoid interpreting
    # onroad-only lateral/angular statistics without their selection coverage.
    result = dict(cell)
    result.pop('files')
    result.update(metrics=report['agent_metrics'], seconds=time.monotonic() - started,
                  native_report=str(folder / 'sd_agent_metrics.json'),
                  native_report_sha256=digest(folder / 'sd_agent_metrics.json'),
                  sampler_audit=str(folder / 'sampler_audit.json'),
                  sampler_audit_sha256=digest(folder / 'sampler_audit.json'),
                  generated_onroad_fraction=report['feature_counts_generated'][1] / max(1, report['num_generated_vehicles']),
                  num_generated_onroad_vehicles=report['feature_counts_generated'][1],
                  num_colliding_vehicles=report['num_colliding_vehicles'],
                  num_generated_vehicles=report['num_generated_vehicles'], num_gt_vehicles=report['num_gt_vehicles'],
                  status='VERIFIED_NATIVE_TEST_N20')
    save(folder / 'result.json', result)
    save(folder / 'verified_completion.json', {'status': 'VERIFIED_COMPLETE', 'training_executed': False,
        'source_cell_sha256': digest(path), 'artifact_sha256': {
            name: digest(folder / name) for name in ('result.json', 'sd_agent_metrics.json', 'sampler_audit.json', 'feature_histograms.json')}})
    print('CC_N20_GUIDANCE_SUCCESS', flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

def plan():
    raw = ROOT / 'output/reports/init_diffusion_657_compact_20261013/live_report.json'
    combined = read(raw)
    selected = next(c for c in combined['comparisons'] if c['source'] == 'main'
                    and c['tier'] == 'long4096' and c['variant'] == 'a2a_k8' and c['budget'] == 10240)
    origins = {seed: next(r for r in selected['native_observations'] if r['seed'] == seed)
               for seed in [4817, 5817]}
    manifest_path = Path(origins[4817]['checkpoint']).parent.parent / 'eval_files.txt'
    files = manifest_path.read_text().splitlines()
    require(len(files) == len(set(files)) == 4096, 'Exact4096 input cohort required')
    file_sha = hashlib.sha256(('\n'.join(files) + '\n').encode()).hexdigest()
    require(all(r['eval_manifest_sha256'] == file_sha for r in origins.values()), 'Both checkpoints must share exact order')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    cells = []
    for option, guidance in OPTIONS.items():
        for seed, origin in origins.items():
            checkpoint = Path(origin['checkpoint'])
            source = checkpoint.parent / 'config.yaml'
            for gen in [2817, 3817]:
                cells.append({'option': option, 'guidance': guidance, 'TRAIN_seed': seed, 'GEN_seed': gen,
                    'steps': 20, 'eps': .05, 'trained_updates': 10240, 'scenes': 4096, 'evaluation_batch_size': 128,
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                    'source_config': str(source), 'source_config_sha256': digest(source),
                    'eval_files_sha256': file_sha, 'files': files, 'sources': sources,
                    'folder': str(RUNS / f'T{seed}_G{gen}_{option}')})
    value = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_report_sha256': digest(raw),
             'deadline': '2026-10-13T12:48:11+08:00', 'options': OPTIONS, 'cells': cells,
             'scope': 'Same10k fresh-trainedK8 checkpoints, twoTRAIN/twoGEN,4096exactmatchedscenes,N20; inference-only guidance; not128epoch proof',
             'selection': 'Lowest mean7ratio to newly reevaluated off anchor with all perTRAIN/pooled L/W/S<=1.15,onroad loss<=.02,vehiclecountGT[.9,1.1]'}
    if (ART / 'protocol.json').exists():
        old = read(ART / 'protocol.json')
        require(old['cells'] == cells, 'Registered protocol differs; never silently regenerate it')
        return old
    save(ART / 'protocol.json', value)
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cell', type=Path)
    args = parser.parse_args()
    if args.cell:
        worker(args.cell)
    p = plan()
    if not args.execute:
        print(json.dumps({'status': 'REGISTERED_NOT_EXECUTED', 'cells': len(p['cells']), 'NFE': 20}))
        return
    deadline = datetime.fromisoformat(p['deadline']).timestamp()
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for i, cell in enumerate(p['cells']):
        folder = Path(cell['folder'])
        if (folder / 'verified_completion.json').exists():
            done = read(folder / 'verified_completion.json')
            require(done['status'] == 'VERIFIED_COMPLETE', 'Invalid completed cell')
            for name, sha in done['artifact_sha256'].items():
                require(digest(folder / name) == sha, 'Actual completed artifact changed')
            rows.append(read(folder / 'result.json'))
            continue
        if time.time() + 180 > deadline:
            save(ART / 'status.json', {'status': 'DEADLINE_NO_NEW_CELL', 'complete_cells': len(rows)})
            break
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        require(not gpu.stdout.strip(), 'Another GPU process is active; do not overlap')
        cell_path = ART / f'cell_{i}.json'
        save(cell_path, cell)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'console.log').open('w') as log:
            proc = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--cell', str(cell_path)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            save(ART / 'status.json', {'status': 'RUNNING', 'pid': proc.pid, 'cell': i, 'option': cell['option'], 'complete_cells': len(rows)})
            try:
                rc = proc.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=30)
                save(ART / 'status.json', {'status': 'DEADLINE_PARTIAL_CELL_EXCLUDED', 'cell': i, 'complete_cells': len(rows)})
                break
        require(rc == 0 and 'CC_N20_GUIDANCE_SUCCESS' in (folder / 'console.log').read_text(),
                'Native cell failed; inspect partial output before retry: ' + str(folder))
        rows.append(read(folder / 'result.json'))
        save(ART / 'observations.json', rows)
    else:
        save(ART / 'status.json', {'status': 'COMPLETE', 'complete_cells': len(rows), 'sampling_steps': 20})
    print(json.dumps({'completed_cells': len(rows), 'planned_cells': len(p['cells'])}))

if __name__ == '__main__':
    main()
