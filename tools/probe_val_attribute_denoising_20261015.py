"""Fixed VAL conditional-denoising errors, genuine96/128 raw/EMA, CPU only."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_mpl_continuous')
import gc;gc.disable()
import argparse,hashlib,json,pickle,random,sys,types
from contextlib import nullcontext
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'output/reports/init_diffusion_continuous_20261014';ART=BASE/'val_attribute_denoising_v1'
PROFILES={'e96':('omega025_e96_20261015',92544),'final128':('omega025_final128_20261015',123392)}
BINS=(0.,.1,.3,.6,1.)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import jinja2,numpy as np,torch,hydra
 from omegaconf import OmegaConf,open_dict
 from tools.runtime_cached_torch_loader_sweep import install_cached_loader
 from tools.checkpoint_weight_audit_20261015 import parameter_digest,checkpoint_parameter_digest
 parser=argparse.ArgumentParser();parser.add_argument('--study',default='val_attribute_denoising_v1');parser.add_argument('--scenes',type=int,default=512);parser.add_argument('--selection',choices=('first_grid','uniform_full_disjoint_v1'),default='first_grid');parser.add_argument('--selection-seed',type=int,default=241016);args=parser.parse_args()
 global ART
 ART=BASE/args.study
 install_cached_loader(torch);torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
 ART.mkdir(exist_ok=False)
 if args.selection=='first_grid':names=json.loads((BASE/'omega025_e104_grid_screen_v2/scenes4096.json').read_text())[:args.scenes]
 else:
  with (BASE/'val_attribute_denoising_v1/val512.pkl').open('rb') as f:excluded=set(pickle.load(f)['files'])
  pool=[name for name in json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text()) if name not in excluded];assert len(pool)==49488
  ids=sorted(np.random.default_rng(args.selection_seed).choice(len(pool),args.scenes,replace=False).tolist());names=[pool[i] for i in ids];assert not excluded.intersection(names)
 assert len(names)==len(set(names))==args.scenes
 manifest=ART/f'val{args.scenes}.pkl'
 with manifest.open('xb') as f:pickle.dump({'files':names},f)
 (ART/'protocol.json').write_text(json.dumps(dict(registered_utc=datetime.now(timezone.utc).isoformat(),VAL_scenes=args.scenes,batch_size=32,source_manifest=str(BASE/('omega025_e104_grid_screen_v2/scenes4096.json' if args.selection=='first_grid' else 'full_sampler_50k_v2/full50000_files.json')),selection=args.selection,selection_seed=args.selection_seed,selection_note='Firstgrid512 covers anearlier segment ofsorteddataset; broader disjoint randomsample checks sensitivity. Bothresultsretained, nooutcomebasedsceneselection',checkpoints=PROFILES,weights=['raw','ema'],time_bins=BINS,seed_per_batch='241015+batchindex',no_training_or_optimization=True,no_CUDA=True,quality_scope='Conditional single-denoise error onVAL bridge samples; not generated50KJSD'),indent=2)+'\n')
 results=[];common_source=None
 for profile,(folder,step) in PROFILES.items():
  snap=BASE/'full_checkpoint_snapshots'/folder;cfg=OmegaConf.load(snap/'config.yaml')
  with open_dict(cfg):
   cfg.model.model_config.initial_map_checkpoint=None;cfg.model.model_config.decoder.init_diffusion.normalizer_path=None;cfg.paths.output_dir=str(ART/'model_output'/profile);cfg.data.scenario_dreamer_eval_set=str(manifest);cfg.data.test_batch_size=32;cfg.data.num_workers=0;cfg.data.pin_memory=False;cfg.data.persistent_workers=False
  model=hydra.utils.instantiate(cfg.model,_recursive_=False).cpu();checkpoint=torch.load(snap/'model.ckpt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'],strict=True);assert checkpoint['global_step']==step
  # Select the native supervised path while keeping every neural module in
  # eval mode: no dropout, no map-stat updates, and no generation rollout.
  model.eval();decoder=model.encoder.init_decoder;decoder.training=True;flow=decoder.G1;assert not flow.model.training and not model.encoder.map_encoder.training
  stats_before={k:v.clone() for k,v in flow.model.named_buffers() if k in ('normal_mean','normal_scale')};objective=flow.component_loss;native_reg=objective.regression;native_loss=flow._component_supervised_loss
  dm=hydra.utils.instantiate(cfg.data);dm.setup('test');assert len(dm.test_dataset)==args.scenes
  for mode in ('raw','ema'):
   sums={};batch_metrics=[];digest=hashlib.sha256();active_agent={}
   def regression(self,name,prediction,target,time,valid,*args,**kwargs):
    answer=native_reg(name,prediction,target,time,valid,*args,**kwargs)
    if name not in ('length','width','velocity'):return answer
    assert prediction.ndim==2 and prediction.shape[1]==1
    t=time.reshape(-1);mask=valid.bool()&(t>0)[:,None]&(t<1)[:,None];error=prediction-target;labels=active_agent['type'];assert len(labels)==len(t)
    batch_metrics[-1][name]=dict(raw=float(answer[0]),weighted=float(answer[1]))
    for stratum in ('all','vehicles'):
     valid_mask=mask if stratum=='all' else mask&(labels==0)[:,None]
     for bucket,(lo,hi) in enumerate(zip(BINS[:-1],BINS[1:])):
      select=valid_mask&(t>=lo)[:,None]&(t<hi)[:,None];values=error[select];pred=prediction[select];key=(name,stratum,bucket);entry=sums.setdefault(key,dict(n=0,sse=0.,sae=0.,signed=0.,negative=0))
      entry['n']+=values.numel();entry['sse']+=float(values.square().sum());entry['sae']+=float(values.abs().sum());entry['signed']+=float(values.sum());entry['negative']+=int((pred<0).sum())
    return answer
   def loss(self,x,x0,noise,time,fixed,agent,prediction=None):
    for value in (x,noise,time,fixed,agent['type']):digest.update(value.detach().contiguous().numpy().tobytes())
    return native_loss(x,x0,noise,time,fixed,agent,prediction)
   objective.regression=types.MethodType(regression,objective);flow._component_supervised_loss=types.MethodType(loss,flow)
   torch.manual_seed(241015);random.seed(241015);np.random.seed(241015)
   context=decoder.ema.average_parameters(flow.parameters()) if mode=='ema' else nullcontext()
   with torch.no_grad(),context:
    expected=checkpoint_parameter_digest(snap/'model.ckpt',mode);assert parameter_digest(flow.named_parameters())==expected
    for index,batch in enumerate(dm.test_dataloader()):
     torch.manual_seed(241015+index);random.seed(241015+index);np.random.seed(241015+index);batch_metrics.append(dict(index=index))
     maps,agents=model.token_processor(batch);active_agent['type']=agents['type'];out=model.encoder(maps,agents)
     assert all(torch.isfinite(v).all() for v in out['initial_logit']);assert set(batch_metrics[-1])=={'index','length','width','velocity'}
     print('VAL_CONDITIONAL',profile,mode,index,flush=True)
    assert parameter_digest(flow.named_parameters())==expected
   assert parameter_digest(flow.named_parameters())==checkpoint_parameter_digest(snap/'model.ckpt','raw')
   for key,value in stats_before.items():torch.testing.assert_close(dict(flow.model.named_buffers())[key],value,rtol=0,atol=0)
   objective.regression=native_reg;flow._component_supervised_loss=native_loss
   if common_source is None:common_source=digest.hexdigest()
   assert common_source==digest.hexdigest(),'VALtarget/source/time/couplingchanged'
   summary=[dict(component=k[0],stratum=k[1],time_lower=BINS[k[2]],time_upper=BINS[k[2]+1],count=v['n'],MSE=v['sse']/max(1,v['n']),MAE=v['sae']/max(1,v['n']),mean_bias=v['signed']/max(1,v['n']),negative_prediction_fraction=v['negative']/max(1,v['n'])) for k,v in sums.items()]
   results.append(dict(profile=profile,weights=mode,step=step,checkpoint_sha256=sha(snap/'model.ckpt'),evaluated_parameter_sha256=expected,source_target_time_sha256=digest.hexdigest(),batches=batch_metrics,errors=summary))
   payload=dict(status='VAL_CONDITIONAL_ATTRIBUTE_PROBE_IN_PROGRESS',updated_utc=datetime.now(timezone.utc).isoformat(),results=results);(ART/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
  del model,checkpoint,dm,flow,decoder,objective
 assert len(results)==4 and not torch.cuda.is_initialized();payload.update(status='VAL_CONDITIONAL_ATTRIBUTE_PROBE_COMPLETE',source_sha256=sha(Path(__file__)),manifest_sha256=sha(manifest),all_native_source_target_time_couplings_exact=True,no_optimizer_updates=True,dropout_disabled=True,TRAIN_normalizers_unchanged=True,CPU_only=True,VAL_scenes=args.scenes,selection=args.selection,limitations='FixedVAL conditional bridge errors, sparse timebins; not full50Krolloutmetrics. Generalization/EMA/integration cause must be checked alongsidepaired generation. No distribution calibration or parameterupdates.');(ART/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(payload['status'],flush=True);os._exit(0)
if __name__=='__main__':main()
