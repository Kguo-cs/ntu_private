"""Prove source-only seed6817 pairing before any full exploratory deployment."""
from pathlib import Path
import os,gc,copy,hashlib,json,sys,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_mpl_continuous');gc.disable()
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
def main():
 import jinja2,torch,hydra
 from hydra import initialize_config_dir,compose
 from omegaconf import OmegaConf,open_dict
 from tools.checkpoint_weight_audit_20261015 import parameter_digest
 from src.smart.diffusion.scale_flow import _noise_endpoint
 torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True);configs=[];models=[];rng=[]
 names=('init_diffusion_657_n24_road_k8_omega025_seed6817_full','init_diffusion_657_n24_road_k8_omega025_empirical_speed_seed6817_full')
 for name in names:
  with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):cfg=compose(config_name='run',overrides=['experiment='+name,'optimizer=default'])
  with open_dict(cfg):cfg.paths.output_dir='/tmp/sim_CDF6817_cpu'
  assert cfg.seed==6817 and cfg.trainer.max_epochs==128 and cfg.data.train_batch_size==512 and cfg.data.num_workers==8
  torch.manual_seed(6817);models.append(hydra.utils.instantiate(cfg.model,_recursive_=False));rng.append(torch.get_rng_state().clone());configs.append(OmegaConf.to_container(cfg,resolve=True))
 def normalized(v):
  if isinstance(v,dict):return {k:normalized(x) for k,x in v.items()}
  if isinstance(v,list):return [normalized(x) for x in v]
  if isinstance(v,str):return v.replace(names[1],names[0])
  return v
 expected=normalized(configs[1]);expected['model']['model_config']['decoder']['init_diffusion']['attribute_source']=configs[0]['model']['model_config']['decoder']['init_diffusion'].get('attribute_source')
 if 'attribute_source' not in configs[0]['model']['model_config']['decoder']['init_diffusion']:expected['model']['model_config']['decoder']['init_diffusion'].pop('attribute_source')
 assert expected==configs[0],'Beyondsource/metadata configurationdiffers'
 baseline,candidate=models;flow=candidate.encoder.init_decoder.G1;reference=baseline.encoder.init_decoder.G1;initial=parameter_digest(flow.named_parameters())
 assert initial==parameter_digest(reference.named_parameters())=='cd5ba829d559df0bb7d50d044977aeb9fa75aeb806ea2f029d482ccfa32af8c1';assert torch.equal(rng[0],rng[1]);assert sum(p.numel() for p in flow.parameters())==8083978
 assert all(torch.equal(v,dict(candidate.named_parameters())[k]) for k,v in baseline.named_parameters());assert all(torch.equal(v,dict(candidate.named_buffers())[k]) for k,v in baseline.named_buffers());assert all(not p.requires_grad for p in candidate.encoder.map_encoder.parameters());assert parameter_digest(candidate.encoder.map_encoder.named_parameters())=='b2b95bdfd833032f2bbaaacf61df8c9d117edb23a877090016f2fa45d1d79be8'
 assert candidate.lr_total_steps==128 and candidate.encoder.init_decoder.ema.num_updates==0
 source=flow.model.attribute_source;assert source.speed and not source.shape and source.knots==149659 and sum(p.numel() for p in source.parameters())==0
 z=torch.randn(100000,7,generator=torch.Generator().manual_seed(241018));state=torch.get_rng_state().clone();raw=_noise_endpoint(reference.model,z,1.);cdf=_noise_endpoint(flow.model,z,1.);assert torch.equal(state,torch.get_rng_state());assert torch.equal(raw[:,:6],cdf[:,:6]) and not torch.equal(raw[:,6],cdf[:,6]);assert torch.isfinite(cdf).all() and (cdf[:,6]>=0).all()
 spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test_init_diffusion_ego_partial.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);x,agent,feature=fixture.InitDiffusionEgoPartialTest().inputs();x=torch.cat((x[:,:6],x[:,6:8].norm(dim=-1,keepdim=True)),-1);agent['expert_input']=x.clone();torch.manual_seed(142015);feature['pt_token']=flow.model.lane_embed(torch.randn(3,128)).detach();out=flow._supervised_loss(x,agent,feature);loss=out[0].mean()+out[1];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None);assert not torch.cuda.is_initialized()
 paths=[Path(__file__),ROOT/'configs/experiment'/('init_diffusion_657_n24_road_k8_omega025_empirical_speed_seed6817_full.yaml'),ROOT/'configs/experiment/init_diffusion_657_n24_road_k8_omega025_empirical_speed_full.yaml',ROOT/'tools/shared_heading_full/model.py',ROOT/'tools/shared_heading_velocity_20261015.py',ROOT/'docs/experiments/init_diffusion_train_attribute_source_full_20261015/quantiles.pt',*sorted((ROOT/'src/smart').rglob('*.py'))]
 result=dict(status='EMPIRICAL_SPEED_SEED6817_SOURCE_ONLY_PAIR_CPU_PASS_NOT_DEPLOYED',seed=6817,parameters=8083978,initial_G1_sha256=initial,initial_map_sha256=parameter_digest(candidate.encoder.map_encoder.named_parameters()),initial_parameters_buffers_CPU_RNG_exact=True,source_only_speed_no_extra_RNG=True,frozen_map=True,EMA_updates=0,source_sha256=source.source_sha256,source_knots=149659,training_epochs=128,LR_horizon_epochs=128,NFE=24,TRAIN_only_source=True,native_backward_finite=True,quality_unproven=True,previous_native8_gate_failed=True,eligibility='Exploratoryonly; must reviewshortlength/lateralregressionsandactualresourceprioritybeforedeployment. CPUproofisnotfullqualityorGPU512proof.',source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths});path=BASE/'empirical_speed_seed6817_cpu_proof_20261015.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}),flush=True);os._exit(0)
if __name__=='__main__':main()
