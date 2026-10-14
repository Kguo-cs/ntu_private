"""CPU TRAIN-only gradient attribution with unchanged native component losses.

Uses raw trained weights and matched TRAIN source/time draws. Small-batch
conditional gradients do not establish the cause of generation-JSD changes.
"""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import gc
gc.disable();gc.collect=lambda *args,**kwargs:0
import hashlib
from pathlib import Path
import pickle
import random
import sys
import types
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf,open_dict

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now,NORMALIZER

ART=ROOT/'output/reports/init_diffusion_overnight_20261014/component_gradient_probe'
OBS=ROOT/'output/reports/init_diffusion_overnight_20261014/recovery/observations.json'
ARMS=('roadK8','size_residual','size_time_width_l1')
BUDGET=4096
SCENES=128


def main():
    torch.set_num_threads(1);ART.mkdir(parents=True,exist_ok=True)
    with (ROOT/'logs/component_loss_scratch/calibration/calibration_files.pkl').open('rb') as f: files=pickle.load(f)
    files=(files['files'] if isinstance(files,dict) else files)[:SCENES]
    assert len(files)==SCENES
    manifest=ART/f'train{SCENES}.pkl'
    with manifest.open('wb') as f:pickle.dump({'files':files},f)
    observations=read(OBS);stats=read(NORMALIZER);results={};hashes=[]
    for arm in ARMS:
        row=next(r for r in observations if (r['variant'],r['seed'],r['budget'],r['eval_seed'])==(arm,4817,BUDGET,2817))
        path=Path(row['checkpoint']);cfg=OmegaConf.load(path.parent/'config.yaml')
        with open_dict(cfg):
            cfg.data.scenario_dreamer_train_sample_list=str(manifest)
            cfg.data.train_batch_size=32;cfg.data.num_workers=0;cfg.data.pin_memory=False
            cfg.data.persistent_workers=False;cfg.data.shuffle=False
        random.seed(241014);np.random.seed(241014);torch.manual_seed(241014)
        model=hydra.utils.instantiate(cfg.model,_recursive_=False).cpu()
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.train()
        flow=model.encoder.init_decoder.G1;objective=flow.component_loss
        for field in ('normal_mean','normal_scale'):
            torch.testing.assert_close(getattr(flow.model,field),torch.tensor(stats[field],dtype=torch.float32),rtol=0,atol=0)
        assert not model.encoder.map_encoder.training
        params=[p for p in flow.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params)==8083978
        dm=hydra.utils.instantiate(cfg.data);dm.setup('fit')
        native_reg=objective.regression;native_cat=objective.categorical
        native_loss=flow._component_supervised_loss
        captured={};digest=hashlib.sha256()
        def regression(instance,name,*args,**kwargs):
            result=native_reg(name,*args,**kwargs)
            assert name not in captured,name
            captured[name]=result[1]
            return result
        def categorical(instance,*args,**kwargs):
            result=native_cat(*args,**kwargs)
            assert 'type' not in captured
            captured['type']=result[1]
            return result
        def loss_hook(instance,x,x0,noise,time,fixed,agent,prediction=None):
            for value in (x,noise,time,fixed):digest.update(value.detach().contiguous().numpy().tobytes())
            return native_loss(x,x0,noise,time,fixed,agent,prediction)
        objective.regression=types.MethodType(regression,objective)
        objective.categorical=types.MethodType(categorical,objective)
        flow._component_supervised_loss=types.MethodType(loss_hook,flow)
        batches=[]
        for i,batch in enumerate(dm.train_dataloader()):
            random.seed(241014+i);np.random.seed(241014+i);torch.manual_seed(241014+i)
            captured.clear()
            maps,agents=model.token_processor(batch)
            result=model.encoder(maps,agents)
            # SMART_GAIL._initial_prediction_loss sums match and collision.
            total=result['initial_logit'][0]+result['initial_logit'][1]
            torch.testing.assert_close(sum(captured.values()),total,rtol=1e-5,atol=1e-7)
            total_g=torch.autograd.grad(total,params,retain_graph=True,allow_unused=True)
            total_squared=sum(g.square().sum() for g in total_g if g is not None)
            assert torch.isfinite(total_squared)
            rows={}
            for name,value in captured.items():
                grad=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                squared=sum(g.square().sum() for g in grad if g is not None)
                dot=sum((g*h).sum() for g,h in zip(grad,total_g) if g is not None and h is not None)
                assert torch.isfinite(squared) and torch.isfinite(dot)
                rows[name]=dict(weighted_loss=float(value.detach()),global_grad_norm=float(squared.sqrt()),
                    cosine_with_total=float(dot/(squared*total_squared).sqrt().clamp_min(1e-30)),
                    dot_over_total_squared=float(dot/total_squared.clamp_min(1e-30)))
                del grad
            assert abs(sum(v['dot_over_total_squared'] for v in rows.values())-1)<2e-4
            batches.append(dict(index=i,agents=len(agents['type']),global_grad_norm=float(total_squared.sqrt()),
                total_loss=float(total.detach()),components=rows))
            print('COMPONENT_GRADIENT',arm,i,flush=True)
            captured.clear();del result,total,total_g,maps,agents,batch
        hashes.append(digest.hexdigest());assert len(set(hashes))==1,'Source/target/time must match'
        results[arm]=dict(checkpoint=str(path),checkpoint_sha256=sha(path),raw_weights=True,
            source_target_time_sha256=digest.hexdigest(),batches=batches)
        save(ART/'results.json',dict(at_utc=now(),script_sha256=sha(__file__),TRAIN_scenes=len(files),batch_size=32,budget=BUDGET,
            normalizer_sha256=sha(NORMALIZER),manifest_sha256=sha(manifest),optimizer_updates=0,results=results,
            caveat=f'CPU raw-weight conditional denoising on{SCENES} TRAIN scenes in32-scene batches. Gradient contribution can be negative or exceed1 due to cancellation. Not actual B512 GPU training gradients, no causal or generation-JSD claim.'))
        del model,flow,objective,dm,checkpoint,params
    assert not torch.cuda.is_initialized()
    os._exit(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget',type=int,default=4096)
    parser.add_argument('--scenes',type=int,default=128)
    parser.add_argument('--arms',default=','.join(ARMS))
    parser.add_argument('--observations',type=Path,default=OBS)
    parser.add_argument('--output',type=Path,default=ART)
    args=parser.parse_args()
    BUDGET=args.budget;SCENES=args.scenes;ARMS=tuple(args.arms.split(','));OBS=args.observations;ART=args.output
    main()
