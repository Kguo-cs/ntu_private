"""Isolated TRAIN collection/fit and native evaluation with physical-size transport.

Evaluation mode accepts the original scratch-sweep CLI after its two arguments;
only a physical output wrapper is installed. Metric calculations stay unchanged.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.init_train_shape_calibration import fit_transport,install_transport


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def setup():
    gc.disable();gc.collect=lambda *unused,**kwargs:0
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    torch.set_num_threads(2)
    return torch


def fit(args):
    torch=setup()
    import hydra
    import lightning as L
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader
    from src.smart.datasets import MultiDataset
    from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
    from tools.scratch_component_loss_sweep import calibrate
    from tools.run_wandb_cc_scratch_20261013 import NORMALIZER
    manifest=ROOT/'logs/component_loss_scratch/calibration/calibration_files.pkl'
    with manifest.open('rb') as f:files=pickle.load(f)['files']
    with (ROOT/'src/waymo_data/full/training_map2_sd_files.pkl').open('rb') as f:full_train=pickle.load(f)['files']
    assert len(files)==8192 and len(set(files))==8192 and set(files)<=set(full_train)
    assert 1<=args.scenes<=8192
    if args.scenes!=8192 and not args.smoke_only:raise ValueError('Registered fit requires8192 TRAIN scenes')
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    selected=folder/'train_files.pkl'
    with selected.open('wb') as f:pickle.dump({'files':files[:args.scenes]},f)
    checkpoint=args.checkpoint.resolve();checkpoint_hash=sha(checkpoint)
    cfg=OmegaConf.load(checkpoint.parent/'config.yaml')
    assert Path(cfg.data.train_raw_dir).resolve()==(ROOT/'src/waymo_data/full/training_map2_sd').resolve()
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    L.seed_everything(4817,workers=True)
    model=hydra.utils.instantiate(cfg.model,_recursive_=False)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict'],strict=True)
    step=int(state['global_step']);del state
    calibrate(model,read(NORMALIZER),verify=True)
    decoder=model.encoder.init_decoder
    assert model.encoder.initial_scene_only and model.n_rollout_closed_val==1 and not model.encoder.sep_map
    assert decoder.sampling_steps==20 and not cfg.model.model_config.decoder.init_diffusion.get('collision_guidance')
    assert not cfg.model.model_config.decoder.init_diffusion.get('sampling_time_grid')
    decoder.use_ema=True
    model.to(args.device).eval()
    dataset=MultiDataset(cfg.data.train_raw_dir,WaymoTargetBuilderVal(),sample_list=str(selected),sample_list_check_exists=False)
    loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0,pin_memory=False)
    L.seed_everything(6817,workers=True)
    generated=[];types=[];egos=[];targets=[];target_types=[];target_egos=[];seen=0
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            batch=batch.to(args.device)
            maps,agents=model.token_processor(batch)
            targets.append(agents['shape'][:,:2].detach().cpu().clone())
            target_types.append(agents['type'].cpu().clone());target_egos.append(agents['ego_mask'].cpu().clone())
            # TRAIN caches have no official full-lane evaluation metadata. Use
            # the native supervised encoder in eval mode: identical map preparation
            # and InitDiffusion EMA sampler, with no official-metric invocation.
            output=model.encoder(maps,agents)['initial_logit']
            assert len(output)==5 and output[3].shape==agents['shape'][:,:2].shape
            generated.append(output[3].cpu())
            types.append(agents['type'].cpu());egos.append(agents['ego_mask'].cpu())
            seen+=int(agents['num_graphs'])
            if (i+1)%8==0:print('TRAIN_COLLECTION',seen,flush=True)
    assert seen==args.scenes and sha(checkpoint)==checkpoint_hash
    tensors=dict(generated=torch.cat(generated),generated_type=torch.cat(types),generated_ego=torch.cat(egos),
                 target=torch.cat(targets),target_type=torch.cat(target_types),target_ego=torch.cat(target_egos))
    torch.save(tensors,folder/'train_sizes.pt')
    metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,global_step=step,
        source_config_sha256=sha(checkpoint.parent/'config.yaml'),TRAIN_scenes=seen,
        original_TRAIN_manifest_sha256=sha(manifest),selected_TRAIN_manifest_sha256=sha(selected),
        train_sizes_sha256=sha(folder/'train_sizes.pt'),normalizer_sha256=sha(NORMALIZER),
        fit_generation_seed=6817,sampling_steps=20,sampling_time_grid=None,collision_guidance=None,
        evaluation_weights='ema',float32_matmul_precision=cfg.float32_matmul_precision,
        fit_device=args.device,smoke_only=args.smoke_only,
        collection_path='native supervised encoder in eval mode, EMA initial decoder; TRAIN graph types retained, no official evaluation metadata invented',
        sources={str(Path(p).relative_to(ROOT)):sha(p) for p in [__file__,ROOT/'tools/init_train_shape_calibration.py']})
    if not args.smoke_only:
        artifact=fit_transport(**tensors)
        artifact['provenance']=metadata
        save(folder/'calibration.json',artifact)
    save(folder/'collection.json',metadata)
    print('TRAIN_SHAPE_COLLECTION_SUCCESS',json.dumps(metadata),flush=True)


def evaluate(args, remaining):
    torch=setup()
    from tools import overnight_init_diffusion_sweep
    sweep=overnight_init_diffusion_sweep.sweep
    artifact=read(args.calibration)
    provenance=artifact['provenance']
    assert not provenance['smoke_only'] and provenance['TRAIN_scenes']==8192
    for p,h in provenance['sources'].items():assert sha(ROOT/p)==h,p
    assert provenance['sampling_steps']==20 and provenance['evaluation_weights']=='ema'
    argv=sweep.parser_for_cli().parse_args(remaining)
    assert argv.evaluation_checkpoints and not argv.variants and not argv.evaluation_sampler
    assert argv.evaluation_weights=='ema' and argv.eval_seed==2817 and argv.extra_eval_seed==3817
    assert argv.eval_batch_size==128 and argv.workers==0
    assert sha(argv.normalizer_json)==provenance['normalizer_sha256']
    checkpoints=read(argv.evaluation_checkpoints)
    assert len(checkpoints)==1 and all(sha(p)==provenance['checkpoint_sha256'] for p in checkpoints.values())
    original=sweep.calibrate;stats=[]
    def hooked(model,*a,**kw):
        result=original(model,*a,**kw)
        if not hasattr(model.encoder.init_decoder,'_train_shape_transport'):
            stats.append(install_transport(model,artifact))
        return result
    sweep.calibrate=hooked
    sys.argv=[sys.argv[0],*remaining]
    sweep.main()
    output=argv.output.resolve();rows=read(output/'summary.json')
    extra=dict(kind='TRAIN_only_monotone_size_postprocessing',calibration_file=str(args.calibration.resolve()),
        calibration_sha256=sha(args.calibration),checkpoint_sha256=provenance['checkpoint_sha256'],
        fit_GEN_seed=6817,fit_TRAIN_scenes=8192,physical_generated_boxes_modified=True,
        evaluator_unchanged=True,fitted_on_evaluation_labels=False,rollout_wrapper_counts=stats)
    for row in rows:
        row['postprocessing']=extra
        metric=read(row['metrics_path']);metric['postprocessing']=extra;save(row['metrics_path'],metric)
    save(output/'summary.json',rows);save(output/'postprocessing.json',extra)
    assert any(s['calls']>0 and s['changed_vehicles']>0 for s in stats)
    print('TRAIN_SHAPE_EVALUATION_SUCCESS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    f=sub.add_parser('fit');f.add_argument('--checkpoint',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--device',choices=['cpu','cuda'],default='cuda');f.add_argument('--scenes',type=int,default=8192)
    f.add_argument('--smoke-only',action='store_true')
    e=sub.add_parser('eval');e.add_argument('--calibration',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    if args.mode=='fit':
        if remaining:parser.error('Unexpected fit arguments')
        fit(args)
    else:evaluate(args,remaining)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
