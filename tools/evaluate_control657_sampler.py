"""Paired native inference ablations on fixed 657 checkpoints.

The 4096-scene grid freezes a sampler choice before any full50k confirmation.
Every option, including N20 anchors, executes the real native TEST path.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_init_diffusion_657_iteration import ART,PYTHON,read,save,sha,score
OUT=ROOT/'logs/control657_sampler_20261012'
OPTIONS={'n20':(20,.05,False),'n40':(40,.05,False),'n80':(80,.05,False),'n40_eps0001':(40,1e-4,False),'hybrid40':(40,.05,True)}
SEEDS=(4817,5817)
GENS=(2817,3817)


def manifest_hash(files):
    return hashlib.sha256(('\n'.join(files)+'\n').encode()).hexdigest()


def child(cell_path):
    gc.disable();gc.collect=lambda *args,**kwargs:0
    os.environ['SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC']='1'
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    from omegaconf import OmegaConf
    from tools.evaluate_init_diffusion_native_sampler import build_eval_config
    from tools.evaluate_init_diffusion_sampler import validate_native_report
    from src.run import run
    torch.set_num_threads(4)
    c=read(cell_path);folder=Path(c['folder']);folder.mkdir(parents=True,exist_ok=True)
    assert sha(c['checkpoint'])==c['checkpoint_sha256']
    assert sha(c['source_config'])==c['source_config_sha256']
    assert manifest_hash(c['files'])==c['eval_files_sha256']
    source=OmegaConf.to_container(OmegaConf.load(c['source_config']),resolve=True)
    manifest=folder/'eval_files.pkl'
    with manifest.open('wb') as handle:pickle.dump({'files':c['files']},handle)
    config=build_eval_config(source,c,folder,manifest)
    config['model']['model_config']['sd_r_reference']=None
    if c['hybrid']:
        callback=config['callbacks']['sampler_audit']
        callback['_target_']='tools.hybrid_position_sampler_hook.make_hybrid_hook'
        callback['hybrid_position']=True
    cfg=OmegaConf.create(config);OmegaConf.save(cfg,folder/'config.yaml')
    started=time.monotonic();run(cfg)
    report=read(folder/'sd_agent_metrics.json');audit=read(folder/'sampler_audit.json')
    validate_native_report(report,audit,c)
    row=dict(c['origin_row']);row.update(report['agent_metrics'])
    row.update(seed=c['TRAIN_seed'],eval_seed=c['GEN_seed'],option=c['option'],variant=c['option'],evaluation_only=True,
        metrics_path=str(folder/'sd_agent_metrics.json'),eval_seconds=time.monotonic()-started,
        eval_manifest_sha256=c['eval_files_sha256'],num_samples=report['num_samples'],
        num_generated_vehicles=report['num_generated_vehicles'],num_gt_vehicles=report['num_gt_vehicles'],
        num_colliding_vehicles=report['num_colliding_vehicles'],
        num_generated_onroad_vehicles=report['feature_counts_generated'][1],
        num_gt_onroad_vehicles=report['feature_counts_gt'][1],
        generated_onroad_fraction=report['feature_counts_generated'][1]/report['num_generated_vehicles'],
        gt_onroad_fraction=report['feature_counts_gt'][1]/report['num_gt_vehicles'])
    save(folder/'result.json',row)
    print('CONTROL657_SAMPLER_SUCCESS',flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)


def make_cells(scenes,options):
    if scenes==4096:
        files=read(ROOT/'tmp/control657_paired_20261012/eval4096_files.json')
    elif scenes==50000:
        files=sorted(p.name for p in (ROOT/'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    else:raise ValueError('Only registered 4096/50000 cohorts are allowed')
    assert len(files)==len(set(files))==scenes
    digest=manifest_hash(files);cells=[]
    for seed in SEEDS:
        folder=ROOT/f'logs/control657_paired_20261012/train{seed}/control657'
        origins=read(folder.parent/'summary.json')
        checkpoint_digest=sha(folder/'last.ckpt');config_digest=sha(folder/'config.yaml')
        for gen in GENS:
            origin=next(r for r in origins if r['budget']==10240 and r['eval_seed']==gen)
            if scenes==4096:assert origin['eval_manifest_sha256']==digest
            for option in options:
                steps,eps,hybrid=OPTIONS[option]
                dest=OUT/(f'T{seed}_G{gen}_{option}' if scenes==4096 else f'full50000/T{seed}_G{gen}_{option}')
                cells.append({'TRAIN_seed':seed,'GEN_seed':gen,'option':option,'steps':steps,'eps':eps,'hybrid':hybrid,
                    'scenes':scenes,'evaluation_batch_size':128,'trained_updates':10240,'checkpoint':str(folder/'last.ckpt'),
                    'checkpoint_sha256':checkpoint_digest,'source_config':str(folder/'config.yaml'),
                    'source_config_sha256':config_digest,'folder':str(dest),'files':files,
                    'eval_files_sha256':digest,'origin_row':origin})
    return cells


def validate_completed(c):
    folder=Path(c['folder']);done=read(folder/'verified_completion.json')
    assert done['status']=='VERIFIED_COMPLETE' and done['returncode']==0
    for name in ('result.json','sd_agent_metrics.json','sampler_audit.json'):
        assert done['artifact_sha256'][name]==sha(folder/name)
    assert done['cell_sha256']==hashlib.sha256(__import__('json').dumps(c,sort_keys=True).encode()).hexdigest()
    row=read(folder/'result.json')
    assert row['num_samples']==c['scenes'] and row['eval_manifest_sha256']==c['eval_files_sha256']
    assert (row['seed'],row['eval_seed'],row['option'])==(c['TRAIN_seed'],c['GEN_seed'],c['option'])
    return row


def execute(cells,label):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC='1')
    rows=[]
    for i,c in enumerate(cells):
        folder=Path(c['folder']);done=folder/'verified_completion.json'
        if done.exists():rows.append(validate_completed(c));continue
        assert not (folder/'config.yaml').exists(),'Inspect incomplete cell before retry'
        cell_path=ART/f'sampler_{label}_cells'/f'{i}.json';save(cell_path,c);folder.mkdir(parents=True,exist_ok=True)
        print('START',folder.name,flush=True)
        with (folder/'console.log').open('w') as log:
            p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--cell',str(cell_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(ART/'sampler_status.json',{'status':'RUNNING','stage':label,'cell':folder.name,'pid':p.pid,'complete_cells':len(rows)})
            rc=p.wait()
        if rc or 'CONTROL657_SAMPLER_SUCCESS' not in (folder/'console.log').read_text():
            save(ART/'sampler_status.json',{'status':'FAILED','stage':label,'cell':folder.name,'returncode':rc});raise RuntimeError('Native sampler cell failed '+folder.name)
        save(done,{'status':'VERIFIED_COMPLETE','returncode':0,
            'cell_sha256':hashlib.sha256(__import__('json').dumps(c,sort_keys=True).encode()).hexdigest(),
            'artifact_sha256':{name:sha(folder/name) for name in ('result.json','sd_agent_metrics.json','sampler_audit.json')}})
        rows.append(validate_completed(c));save(ART/f'sampler_{label}_observations.json',rows)
        print('DONE',folder.name,flush=True)
    return rows


def compare(rows,options):
    anchors=[r for r in rows if r['option']=='n20']
    return {option:score([r for r in rows if r['option']==option],anchors) for option in options}


def full_confirmation():
    frozen=read(ART/'sampler_frozen_selection.json');comparison=ART/'sampler_comparison.json'
    assert frozen['selection_is_frozen'] is True and frozen['grid_comparison_sha256']==sha(comparison)
    grid=read(comparison);selected=frozen['selected_option']
    assert grid['status']=='COMPLETE' and grid['selected']==selected
    assert selected!='n20','N20 is unchanged; no improved sampler to confirm'
    for c in read(ART/'sampler_protocol.json')['cells']:validate_completed(c)
    cells=make_cells(50000,['n20',selected])
    protocol={'frozen_selection_sha256':sha(ART/'sampler_frozen_selection.json'),'cells':cells,
        'selection_on_full_set':False,'driver_sha256':sha(__file__),'created_utc':datetime.now(timezone.utc).isoformat()}
    path=ART/'sampler_full_protocol.json'
    if path.exists():assert read(path)['driver_sha256']==sha(__file__) and read(path)['frozen_selection_sha256']==protocol['frozen_selection_sha256']
    else:save(path,protocol)
    rows=execute(cells,'full');results=compare(rows,['n20',selected]);s=results[selected]
    save(ART/'sampler_full_comparison.json',{'selected':selected,'comparisons':results,'status':'COMPLETE',
        'beats657':s['R']<1 and s['all_TRAIN_gates_pass'],'frozen_selection_sha256':protocol['frozen_selection_sha256']})
    save(ART/'sampler_status.json',{'status':'COMPLETE','stage':'full','selected':selected,'beats657':s['R']<1 and s['all_TRAIN_gates_pass']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--cell',type=Path)
    parser.add_argument('--full-confirmation',action='store_true');parser.add_argument('--confirm-selected',action='store_true')
    parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
    if args.cell:child(args.cell);return
    if args.wait_pid and args.execute:
        from tools.run_init_diffusion_657_round2 import alive
        save(ART/'sampler_status.json',{'status':'WAITING_FOR_SERIAL_GPU','wait_pid':args.wait_pid,'pid':os.getpid()})
        while alive(args.wait_pid):time.sleep(5)
        prior=read(ART/'round2_status.json')
        assert prior['status'].startswith('COMPLETE'),'Inspect round2 failure before starting sampler'
    if args.full_confirmation:
        assert args.execute,'Full confirmation must execute the frozen decision'
        full_confirmation();return
    cells=make_cells(4096,list(OPTIONS))
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'definition':'fresh nativeN20 anchors; equalTRAIN means of GEN means; sampler only, no training',
        'cells':cells,'options':OPTIONS,'scope':'4096 matched scenes, bothTRAIN, bothGEN, same fixed657 weights',
        'selection':'lowestR among options passing pooled and bothTRAIN gates; full50k frozen before confirmation, including freshN20 full anchors',
        'driver_sha256':sha(__file__)}
    path=ART/'sampler_protocol.json'
    if path.exists():assert read(path)['driver_sha256']==sha(__file__)
    else:save(path,plan)
    if not args.execute:print('Planned native sampler cells:',len(cells));return
    rows=execute(cells,'grid');results=compare(rows,list(OPTIONS))
    eligible=[o for o,s in results.items() if s['all_TRAIN_gates_pass']]
    selected=min(eligible,key=lambda o:results[o]['R'])
    path=ART/'sampler_comparison.json'
    save(path,{'comparisons':results,'selected':selected,'status':'COMPLETE','N_model_calls':{k:v[0] for k,v in OPTIONS.items()}})
    save(ART/'sampler_frozen_selection.json',{'selected_option':selected,'selection_is_frozen':True,
        'grid_comparison_sha256':sha(path),'grid_observations_sha256':sha(ART/'sampler_grid_observations.json'),
        'created_utc':datetime.now(timezone.utc).isoformat(),'expected_full_cells':8 if selected!='n20' else 0})
    save(ART/'sampler_status.json',{'status':'COMPLETE','stage':'grid','selected':selected,'complete_cells':len(rows)})
    if args.confirm_selected and selected!='n20':full_confirmation()


if __name__=='__main__':main()
