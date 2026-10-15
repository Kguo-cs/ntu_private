"""Real native-recipe proof for the speed-positivity ablation."""
from pathlib import Path
import gc,hashlib,json,sys,copy,importlib.util,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));gc.disable()

def main():
 import jinja2,hydra,torch
 from hydra import initialize_config_dir,compose
 from omegaconf import OmegaConf,open_dict
 from src.smart.diffusion.scale_flow import _noise_endpoint
 from tools.run_native_prefix_training_20261015 import digest_parameters
 from tools.rectified_speed_full.model import RectifiedGaussianSpeedSource
 torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
 models=[];rngs=[]
 for name in ('init_diffusion_omega025_native8','init_diffusion_omega025_rectified_speed_native8'):
  with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):cfg=compose(config_name='run',overrides=['experiment='+name,'optimizer=default'])
  with open_dict(cfg):cfg.paths.output_dir='/tmp/sim_rectified_speed_cpu'
  torch.manual_seed(4817);models.append(hydra.utils.instantiate(cfg.model,_recursive_=False));rngs.append(torch.get_rng_state().clone())
 baseline,candidate=models;decoder=candidate.encoder.init_decoder;flow=decoder.G1
 assert torch.equal(*rngs)
 assert digest_parameters(baseline.encoder.init_decoder.G1)==digest_parameters(flow)
 assert digest_parameters(baseline.encoder.map_encoder)==digest_parameters(candidate.encoder.map_encoder)
 assert sum(p.numel() for p in flow.parameters())==8083978
 z=torch.randn(100000,7,generator=torch.Generator().manual_seed(241015));before=torch.get_rng_state().clone()
 raw=_noise_endpoint(baseline.encoder.init_decoder.G1.model,z,1.);fixed=_noise_endpoint(flow.model,z,1.)
 assert torch.equal(before,torch.get_rng_state()) and torch.equal(fixed[:,:6],raw[:,:6])
 assert torch.equal(fixed[:,6],raw[:,6].clamp_min(0))
 source=RectifiedGaussianSpeedSource();source.load_state_dict(flow.model.attribute_source.state_dict(),strict=True)
 assert torch.equal(source(z,raw),fixed)
 try:decoder.set_extra_state(baseline.encoder.init_decoder.get_extra_state())
 except ValueError:wrong_source_rejected=True
 else:raise AssertionError('Gaussian checkpoint silently changed source')
 spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test_init_diffusion_ego_partial.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 x,agent,features=fixture.InitDiffusionEgoPartialTest().inputs();x=torch.cat((x[:,:6],x[:,6:8].norm(dim=-1,keepdim=True)),-1);agent['expert_input']=x.clone()
 torch.manual_seed(142015);features['pt_token']=flow.model.lane_embed(torch.randn(3,128)).detach()
 torch.manual_seed(142017);result=flow._supervised_loss(x,agent,features);loss=result[0].mean()+result[1];assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None)
 files=[Path(__file__),ROOT/'tools/rectified_speed_full/model.py',ROOT/'tools/rectified_speed_full/__init__.py',ROOT/'configs/experiment/init_diffusion_657_n24_road_k8_omega025_rectified_speed_full.yaml',ROOT/'configs/experiment/init_diffusion_omega025_rectified_speed_native8.yaml']
 files+=sorted((ROOT/'src/smart').rglob('*.py'))
 proof=dict(status='RECTIFIED_SPEED_NATIVE_FULL_CONFIG_CPU_PASS_NOT_TRAINED',parameters=8083978,initial_G1_sha256=digest_parameters(flow),initial_map_sha256=digest_parameters(candidate.encoder.map_encoder),initial_parameters_and_CPU_RNG_exact=True,only_speed_changed=True,no_extra_RNG=True,finite_native_backward=True,source_checkpoint_exact=True,wrong_source_rejected=wrong_source_rejected,
  zero_source_fraction=float((fixed[:,6]==0).float().mean()),gaussian_negative_fraction=float((raw[:,6]<0).float().mean()),quality_unproven=True,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
 out=ROOT/'output/reports/init_diffusion_continuous_20261014/rectified_speed_native_cpu_proof_20261015.json';out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='source_hashes'}),flush=True);os._exit(0)

if __name__=='__main__':main()
