"""CPU TRAIN-only gradient attribution with unchanged native component losses.

Uses raw orEMA trained weights and matched TRAIN source/time draws. Small-batch
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
NATIVE_PREFIX_STUDY=None
FULL_CHECKPOINT=None
FULL_CONFIG=None
USE_EMA=False
PROBE_OUTPUT_COMPONENTS=False


def main():
    torch.set_num_threads(1);ART.mkdir(parents=True,exist_ok=True)
    with (ROOT/'logs/component_loss_scratch/calibration/calibration_files.pkl').open('rb') as f: files=pickle.load(f)
    files=(files['files'] if isinstance(files,dict) else files)[:SCENES]
    assert len(files)==SCENES
    manifest=ART/f'train{SCENES}.pkl'
    with manifest.open('wb') as f:pickle.dump({'files':files},f)
    observations=[] if FULL_CHECKPOINT is not None else read(OBS if NATIVE_PREFIX_STUDY is None else NATIVE_PREFIX_STUDY/'observations.json');stats=read(NORMALIZER);results={};hashes=[]
    for arm in ARMS:
        if FULL_CHECKPOINT is not None:
            assert len(ARMS)==1 and FULL_CONFIG is not None
            row=dict(checkpoint=str(FULL_CHECKPOINT))
            config_path=FULL_CONFIG
            OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
        elif NATIVE_PREFIX_STUDY is None:
            row=next(r for r in observations if (r['variant'],r['seed'],r['budget'],r['eval_seed'])==(arm,4817,BUDGET,2817))
            config_path=Path(row['checkpoint']).parent/'config.yaml'
        else:
            row=next(r for r in observations if r['arm']==arm)
            assert row['training_audit']['actual_step']==BUDGET
            config_path=Path(row['report']).parent/'.hydra/config.yaml'
            OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
        path=Path(row['checkpoint']);cfg=OmegaConf.load(config_path)
        with open_dict(cfg):
            if NATIVE_PREFIX_STUDY is not None or FULL_CHECKPOINT is not None:
                cfg.paths.output_dir=str(ART/'portable_output'/arm)
                cfg.paths.log_dir=str(ART/'portable_output'/arm)
            if FULL_CHECKPOINT is not None:
                cfg.model.model_config.initial_map_checkpoint=None
                cfg.model.model_config.decoder.init_diffusion.normalizer_path=None
            cfg.data.scenario_dreamer_train_sample_list=str(manifest)
            cfg.data.train_batch_size=32;cfg.data.num_workers=0;cfg.data.pin_memory=False
            cfg.data.persistent_workers=False;cfg.data.shuffle=False
        random.seed(241014);np.random.seed(241014);torch.manual_seed(241014)
        model=hydra.utils.instantiate(cfg.model,_recursive_=False).cpu()
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.train()
        if FULL_CHECKPOINT is not None:
            assert checkpoint['global_step']==BUDGET
            provenance=read(path.with_name('provenance.json'))
            assert provenance['sha256']==sha(path) and provenance['global_step']==BUDGET
            # Portable eval configs omit the AIRL initialization path. Their
            # loaded map weights are correct, but model.train() alone therefore
            # enables map dropout. Restore this experiment's frozen-map mode
            # explicitly; never reload a different map or optimize this model.
            model.encoder.map_encoder.requires_grad_(False)
            model.encoder.map_encoder.eval()
        if USE_EMA:
            assert model.encoder.init_decoder.ema.num_updates==BUDGET
            model.encoder.init_decoder.ema.copy_to(model.encoder.init_decoder.G1.parameters())
        flow=model.encoder.init_decoder.G1;objective=flow.component_loss
        normalizer_before={field:getattr(flow.model,field).clone() for field in ('normal_mean','normal_scale')}
        for field in ('normal_mean','normal_scale'):
            expected=(torch.tensor(stats[field],dtype=torch.float32) if NATIVE_PREFIX_STUDY is None and FULL_CHECKPOINT is None else
                      checkpoint['state_dict']['encoder.init_decoder.G1.model.'+field])
            torch.testing.assert_close(getattr(flow.model,field),expected,rtol=0,atol=0)
        assert not model.encoder.map_encoder.training
        params=[p for p in flow.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params)==8083978
        shared_omega=(flow.heading_objective=='angular_velocity' and
            model.encoder.init_decoder.denoiser_options.get('heading_velocity_readout','').startswith('shared_state_row3'))
        output=flow.model.to_out_m_delta.mlp[-1]
        head_ids={id(output.weight),id(output.bias)}
        heading_output_ids={id(output.weight),id(output.bias)} if shared_omega else set()
        def heading_row_inner(first,second):
            values=((a[3]*b[3]).sum() for p,a,b in zip(params,first,second)
                if id(p) in heading_output_ids and a is not None and b is not None)
            return sum(values,params[0].new_zeros(()))
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
            heading_g=torch.autograd.grad(captured['heading'],params,retain_graph=True,allow_unused=True)
            heading_squared=sum(g.square().sum() for g in heading_g if g is not None)
            heading_row_squared=heading_row_inner(heading_g,heading_g) if shared_omega else None
            total_squared=sum(g.square().sum() for g in total_g if g is not None)
            assert torch.isfinite(total_squared)
            rows={}
            output_vectors={key:{} for key in ('length','width','velocity')} if PROBE_OUTPUT_COMPONENTS else None
            for name,value in captured.items():
                grad=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                squared=sum(g.square().sum() for g in grad if g is not None)
                dot=sum((g*h).sum() for g,h in zip(grad,total_g) if g is not None and h is not None)
                heading_dot=sum((g*h).sum() for g,h in zip(grad,heading_g) if g is not None and h is not None)
                assert torch.isfinite(squared) and torch.isfinite(dot)
                rows[name]=dict(weighted_loss=float(value.detach()),global_grad_norm=float(squared.sqrt()),
                    cosine_with_total=float(dot/(squared*total_squared).sqrt().clamp_min(1e-30)),
                    cosine_with_heading=float(heading_dot/(squared*heading_squared).sqrt().clamp_min(1e-30)),
                    dot_over_total_squared=float(dot/total_squared.clamp_min(1e-30)))
                if shared_omega:
                    row_squared=heading_row_inner(grad,grad)
                    row_dot=heading_row_inner(grad,heading_g)
                    rows[name].update(omega_output_row_grad_norm=float(row_squared.sqrt()),
                        cosine_with_heading_output_row=float(row_dot/(row_squared*heading_row_squared).sqrt().clamp_min(1e-30)),
                        projection_on_heading_output_row=float(row_dot/heading_row_squared.clamp_min(1e-30)))
                if PROBE_OUTPUT_COMPONENTS:
                    for key,index in (('length',4),('width',5),('velocity',6)):
                        values=[g[index].reshape(-1) for p,g in zip(params,grad) if id(p) in head_ids and g is not None]
                        output_vectors[key][name]=(torch.cat(values).detach() if values else params[0].new_zeros(output.in_features+1))
                del grad
            assert abs(sum(v['dot_over_total_squared'] for v in rows.values())-1)<2e-4
            item=dict(index=i,agents=len(agents['type']),global_grad_norm=float(total_squared.sqrt()),
                total_loss=float(total.detach()),components=rows)
            if PROBE_OUTPUT_COMPONENTS:
                interactions={}
                for key,vectors in output_vectors.items():
                    own=vectors[key];collision=vectors['collision'];own2=own.square().sum();col2=collision.square().sum();inner=(own*collision).sum()
                    interactions[key]=dict(own_weighted_head_grad_norm=float(own2.sqrt()),
                        collision_weighted_head_grad_norm=float(col2.sqrt()),
                        cosine_collision_with_own=float(inner/(own2*col2).sqrt().clamp_min(1e-30)),
                        collision_projection_on_own=float(inner/own2.clamp_min(1e-30)),
                        own_bias_derivative=float(own[-1]),collision_bias_derivative=float(collision[-1]))
                item['physical_output_row_interactions']=interactions
                del output_vectors
            batches.append(item)
            print('COMPONENT_GRADIENT',arm,i,flush=True)
            captured.clear();del result,total,total_g,heading_g,maps,agents,batch
        hashes.append(digest.hexdigest());assert len(set(hashes))==1,'Source/target/time must match'
        for field,value in normalizer_before.items():
            torch.testing.assert_close(getattr(flow.model,field),value,rtol=0,atol=0)
        results[arm]=dict(checkpoint=str(path),checkpoint_sha256=sha(path),raw_weights=not USE_EMA,
            EMA_weights=USE_EMA,denoiser_training_mode=flow.model.training,
            frozen_map_eval=True,checkpoint_normalizer_unchanged=True,
            source_target_time_sha256=digest.hexdigest(),batches=batches)
        save(ART/'results.json',dict(at_utc=now(),script_sha256=sha(__file__),TRAIN_scenes=len(files),batch_size=32,budget=BUDGET,
            normalizer_sha256=sha(NORMALIZER),manifest_sha256=sha(manifest),optimizer_updates=0,results=results,
            native_prefix_study=None if NATIVE_PREFIX_STUDY is None else str(NATIVE_PREFIX_STUDY),
            native_checkpoint_stats_verified=NATIVE_PREFIX_STUDY is not None or FULL_CHECKPOINT is not None,
            full_checkpoint=None if FULL_CHECKPOINT is None else str(FULL_CHECKPOINT),
            physical_output_row_interactions_enabled=PROBE_OUTPUT_COMPONENTS,
            config_sha256=sha(config_path),
            caveat=f'CPU {"EMA" if USE_EMA else "raw"}-weight conditional denoising with training-mode dropout on{SCENES} TRAIN scenes in32-scene batches. Gradient contribution can be negative or exceed1 due to cancellation. Not actual B512 GPU training gradients, no causal or generation-JSD claim.'))
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
    parser.add_argument('--native-prefix-study',type=Path)
    parser.add_argument('--full-checkpoint',type=Path)
    parser.add_argument('--full-config',type=Path)
    parser.add_argument('--weights',choices=('raw','ema'),default='raw')
    parser.add_argument('--output-head-components',action='store_true')
    args=parser.parse_args()
    BUDGET=args.budget;SCENES=args.scenes;ARMS=tuple(args.arms.split(','));OBS=args.observations;ART=args.output
    NATIVE_PREFIX_STUDY=args.native_prefix_study
    FULL_CHECKPOINT=args.full_checkpoint;FULL_CONFIG=args.full_config;USE_EMA=args.weights=='ema'
    PROBE_OUTPUT_COMPONENTS=args.output_head_components
    assert not (NATIVE_PREFIX_STUDY is not None and FULL_CHECKPOINT is not None)
    assert (FULL_CONFIG is not None)==(FULL_CHECKPOINT is not None)
    main()
