"""Paired native N20 sampler/guidance evaluation after the evening K8 fit.

Fit code and checkpoints are immutable. Screening uses4096 matched scenes;
the selected inference option and its off anchor are rechecked on full50K.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_cc_n20_guidance as native
from tools import resume_init_diffusion_after_sampler_merge_20261013 as training

ART = ROOT / 'output/reports/init_diffusion_evening_20261013/n20_sampler'
RUNS = ROOT / 'logs/init_diffusion_evening_20261013/n20_sampler'
END = datetime.fromisoformat('2026-10-13T22:30:00+08:00').timestamp()
read, save, sha = native.read, native.save, native.digest
FOCUS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd')
JSD = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')
OFF = native.OPTIONS['off']
OPTIONS = {
    'off_uniform': {'guidance': OFF, 'time_grid': {}},
    'guide_light': {'guidance': native.OPTIONS['light'], 'time_grid': {}},
    'guide_cc': {'guidance': native.OPTIONS['cc'], 'time_grid': {}},
    'power075_terminal005': {'guidance': OFF, 'time_grid': {'power': .75, 'terminal_time': .05}},
    'power15_terminal005': {'guidance': OFF, 'time_grid': {'power': 1.5, 'terminal_time': .05}},
    'cosine_terminal005': {'guidance': OFF, 'time_grid': {'kind': 'cosine', 'terminal_time': .05}},
    'power15_raw': {'guidance': OFF, 'time_grid': {'power': 1.5}},
    'cosine_raw': {'guidance': OFF, 'time_grid': {'kind': 'cosine'}},
}
STOP = False
original_hook = native.make_hook


def make_grid_hook(options, time_grid, **kwargs):
    base = original_hook(options=options, **kwargs)
    class Hook(type(base)):
        def on_test_start(self, trainer, model):
            super().on_test_start(trainer, model)
            from src.smart.diffusion.sampling_time_grid import build_sampling_time_grid, parse_sampling_time_grid
            import torch
            flow = model.encoder.init_decoder.G1
            self.old_time_grid = dict(flow.sampling_time_grid)
            flow.sampling_time_grid = parse_sampling_time_grid(time_grid)
            self.record['sampling_time_grid'] = dict(flow.sampling_time_grid)
            self.record['actual_time_nodes'] = build_sampling_time_grid(
                20, device='cpu', dtype=torch.float32, config=flow.sampling_time_grid).tolist()
            save(self.audit_path, self.record)

        def on_test_end(self, trainer, model):
            model.encoder.init_decoder.G1.sampling_time_grid = self.old_time_grid
            super().on_test_end(trainer, model)
    return Hook(**kwargs)


def worker(path):
    cell = read(path)
    # Native worker builds and strictly restores the unchanged checkpoint model;
    # only the TEST callback overrides the inference attributes after restore.
    def hook(options, **kwargs):
        return make_grid_hook(options, time_grid=cell['time_grid'], **kwargs)
    native.make_hook = hook
    native.worker(path)


def source_hashes():
    paths = (*native.SOURCES, 'src/smart/diffusion/sampling_time_grid.py',
             'src/smart/model/smart.py', 'tools/evaluate_init_n20_evening_20261013.py')
    return {p: sha(ROOT / p) for p in paths}


def register():
    if (ART / 'protocol.json').exists():
        return read(ART / 'protocol.json')
    value = dict(registered_utc=datetime.now().astimezone().isoformat(), deadline=END,
                 sources=source_hashes(), options=OPTIONS, TRAIN=[4817], GEN=[2817, 3817],
                 screening_scenes=4096, final_scenes=50000, sampling_steps=20,
                 denoiser_parameter_cap=8100000, fit_unchanged=True,
                 primary='Mean speed/angle/length JSD ratios versus same-arm off_uniform.',
                 eligibility='Both GEN complete; each focused ratio<=1.05, width/near<=1.15, lateral absolute rise<=.02 or ratio<=1.15, onroad decline<=.02, generated count within1% of anchor; focused mean<=.98.',
                 full_rule='On full50K compare same frozen option to off anchor; no change of option after viewing full results. Each GEN reported, plus pooled histogram JSD. Final127 iu3 reference is descriptive, not equal-budget control.',
                 clamp_diagnostic='Raw cosine/power can end below training t_eps=.05; terminal-controlled grids retain the same final model time. Training objective/weights never changed.')
    save(ART / 'protocol.json', value)
    return value


def compare(rows, arm, scenes):
    subset = [r for r in rows if r['arm'] == arm and r['scenes'] == scenes]
    base = [r for r in subset if r['option'] == 'off_uniform']
    if len(base) != 2:
        return []
    result = []
    for option in OPTIONS:
        rr = [r for r in subset if r['option'] == option]
        if len(rr) != 2:
            continue
        ratios = {k: statistics.fmean(r['metrics'][k] for r in rr) /
                  statistics.fmean(r['metrics'][k] for r in base) for k in JSD + ('collision_rate',)}
        onroad = statistics.fmean(r['generated_onroad_fraction'] for r in rr)
        base_onroad = statistics.fmean(r['generated_onroad_fraction'] for r in base)
        lat = statistics.fmean(r['metrics']['lat_dev_jsd'] for r in rr)
        base_lat = statistics.fmean(r['metrics']['lat_dev_jsd'] for r in base)
        count = statistics.fmean(r['num_generated_vehicles'] for r in rr) / statistics.fmean(r['num_generated_vehicles'] for r in base)
        focus = statistics.fmean(ratios[k] for k in FOCUS)
        gates = (all(ratios[k] <= 1.05 for k in FOCUS)
                 and all(ratios[k] <= 1.15 for k in ('nearest_dist_jsd', 'width_jsd'))
                 and (lat - base_lat <= .02 or ratios['lat_dev_jsd'] <= 1.15)
                 and onroad >= base_onroad - .02 and .99 <= count <= 1.01)
        result.append(dict(arm=arm, option=option, scenes=scenes, ratios=ratios,
                           R_focus=focus, R_jsd=statistics.fmean(ratios[k] for k in JSD),
                           R_attr=statistics.fmean(ratios[k] for k in ('length_jsd', 'width_jsd', 'speed_jsd')),
                           onroad_fraction=onroad, onroad_delta=onroad-base_onroad,
                           count_ratio=count, gates_pass=gates,
                           eligible=option != 'off_uniform' and gates and focus <= .98))
    return result


def cell(arm, origin, gen, option, files, phase):
    cp = Path(origin['checkpoint'])
    config = cp.parent / 'config.yaml'
    return dict(arm=arm, option=option, phase=phase, **copy.deepcopy(OPTIONS[option]),
                TRAIN_seed=4817, GEN_seed=gen, steps=20, eps=.05, trained_updates=16384,
                scenes=len(files), evaluation_batch_size=128,
                checkpoint=str(cp), checkpoint_sha256=sha(cp),
                source_config=str(config), source_config_sha256=sha(config),
                files=files, eval_files_sha256=__import__('hashlib').sha256(('\n'.join(files)+'\n').encode()).hexdigest(),
                sources=source_hashes(), folder=str(RUNS / phase / f'{arm}_G{gen}_{option}'))


def run_cell(value, rows):
    global STOP
    folder = Path(value['folder'])
    if (folder / 'verified_completion.json').exists():
        proof = read(folder / 'verified_completion.json')
        assert proof['status'] == 'VERIFIED_COMPLETE'
        for name, digest in proof['artifact_sha256'].items():
            assert sha(folder / name) == digest
        rows.append(read(folder / 'result.json'))
        return True
    if STOP or time.time() + (300 if value['scenes'] == 50000 else 60) >= END:
        return False
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
    assert not gpu.stdout.strip(), 'Another GPU process is active'
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'cell.json'
    save(path, value)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    with (folder / 'console.log').open('w') as log:
        proc = subprocess.Popen([native.PYTHON, __file__, '--cell', str(path)], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            save(ART / 'status.json', dict(status='RUNNING_NATIVE_TEST', arm=value['arm'], option=value['option'],
                                         scenes=value['scenes'], GEN=value['GEN_seed'], child_pid=proc.pid,
                                         completed_cells=len(rows), updated_local=datetime.now().astimezone().isoformat()))
            if STOP or time.time() >= END:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                save(ART / 'status.json', dict(status='PARTIAL_CELL_EXCLUDED', folder=str(folder), completed_cells=len(rows)))
                return False
            time.sleep(10)
    assert proc.returncode == 0 and 'CC_N20_GUIDANCE_SUCCESS' in (folder / 'console.log').read_text(), str(folder)
    result = read(folder / 'result.json')
    audit = read(result['sampler_audit'])
    assert audit['denoiser_parameters'] == 8083978 and set(audit['observed_NFE_per_sample']) == {20}
    assert result['scenes'] == result['num_samples'] if 'num_samples' in result else result['scenes'] == value['scenes']
    assert len(audit['actual_time_nodes']) == 21
    report = read(result['native_report'])
    assert report['decoder']['sampling_time_grid'] == audit['sampling_time_grid']
    rows.append(result)
    save(ART / 'observations.json', rows)
    save(ART / 'comparisons.json', [c for arm in ('fresh657', 'freshK8')
                                   for n in (4096, 50000) for c in compare(rows, arm, n)])
    print(json.dumps(dict(arm=value['arm'], option=value['option'], GEN=value['GEN_seed'], scenes=value['scenes'],
                         focus={k: result['metrics'][k] for k in FOCUS})), flush=True)
    return True


def execute():
    global STOP
    def halt(*unused):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    plan = register()
    assert plan['sources'] == source_hashes()
    while not STOP and time.time() < END:
        status = read(training.ART / 'status.json')
        if status['status'] == 'COMPLETE_LONG_TRAIN':
            break
        if status['status'] not in ('REGISTERED', 'RUNNING'):
            raise RuntimeError('Training did not finish: ' + str(status))
        save(ART / 'status.json', dict(status='WAITING_FOR_REGISTERED_TRAIN', training_status=status['status']))
        time.sleep(10)
    if STOP or time.time() >= END:
        return
    endpoints = read(training.ART / 'observations.json')
    origins = {arm: next(r for r in endpoints if r['variant'] == arm and r['budget'] == 16384
                         and not r['evaluation_only']) for arm in ('fresh657', 'freshK8')}
    files = read(ROOT / 'tmp/control657_paired_20261012/eval4096_files.json')
    assert len(files) == len(set(files)) == 4096
    save(ART / 'frozen_endpoints.json', origins)
    rows = []
    # Separate native processes reset GEN after strict restore; no state leaks.
    for option in OPTIONS:
        for arm, origin in origins.items():
            for gen in (2817, 3817):
                if not run_cell(cell(arm, origin, gen, option, files, 'screen4096'), rows):
                    return
    decisions = {}
    for arm in origins:
        eligible = [c for c in compare(rows, arm, 4096) if c['eligible']]
        decisions[arm] = min(eligible, key=lambda c: c['R_focus'])['option'] if eligible else 'off_uniform'
    save(ART / 'frozen_full50k_decision.json', dict(options=decisions, observations_sha256=sha(ART / 'observations.json'),
                                                selection_complete_before_full=True))
    files50 = sorted(p.name for p in (ROOT / 'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    assert len(files50) == len(set(files50)) == 50000
    for arm, origin in origins.items():
        for option in dict.fromkeys(('off_uniform', decisions[arm])):
            for gen in (2817, 3817):
                if not run_cell(cell(arm, origin, gen, option, files50, 'full50000'), rows):
                    return
    save(ART / 'status.json', dict(status='COMPLETE', completed_cells=len(rows),
                                 full128epoch_verified=False, NFE=20, decisions=decisions))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cell', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.cell:
        worker(args.cell)
    elif args.execute:
        execute()
    else:
        register()
