"""CPU proof of raw/EMA selection using genuine104epoch trained weights."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import gc;gc.disable()
import copy,hashlib,importlib.util,json,sys
from contextlib import nullcontext
from datetime import datetime,timezone
from types import MethodType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'

def main():
 import jinja2,torch,hydra
 from omegaconf import OmegaConf,open_dict
 from tools.checkpoint_weight_audit_20261015 import parameter_digest,checkpoint_parameter_digest
 torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
 snapshot=BASE/'full_checkpoint_snapshots/omega025_e104_20261015';cfg=OmegaConf.load(snapshot/'config.yaml')
 with open_dict(cfg):cfg.model.model_config.decoder.init_diffusion.normalizer_path=None;cfg.paths.output_dir='/tmp/sim_raw_ema_cpu'
 model=hydra.utils.instantiate(cfg.model,_recursive_=False);checkpoint=torch.load(snapshot/'model.ckpt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();decoder=model.encoder.init_decoder;flow=decoder.G1
 assert checkpoint['global_step']==decoder.ema.num_updates==100256
 expected={mode:checkpoint_parameter_digest(snapshot/'model.ckpt',mode) for mode in ('raw','ema')};assert expected['raw']!=expected['ema']
 original_prep,original_map,original_infer=decoder._prepare_ego_context,decoder._initial_map_feature,decoder._infer
 decoder._prepare_ego_context=MethodType(lambda self,agent:(None,None,None,None),decoder)
 decoder._initial_map_feature=MethodType(lambda self,*args:None,decoder)
 decoder._infer=MethodType(lambda self,*args:parameter_digest(self.G1.named_parameters()),decoder)
 context_results={}
 for mode in ('raw','ema'):
  decoder.use_ema=mode=='ema'
  with torch.no_grad():observed=decoder({})
  assert observed==expected[mode];assert parameter_digest(flow.named_parameters())==expected['raw']
  context_results[mode]=observed
 decoder._prepare_ego_context,decoder._initial_map_feature,decoder._infer=original_prep,original_map,original_infer
 decoder.use_ema=True
 spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test_init_diffusion_ego_partial.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 _,agent,features=fixture.InitDiffusionEgoPartialTest().inputs();decoder._prepare_ego_context(agent)
 torch.manual_seed(32415);lane_input=torch.randn(3,128);outputs={};sources={};rng={};calls=[0]
 hook=flow.model.register_forward_pre_hook(lambda *unused:calls.__setitem__(0,calls[0]+1))
 for mode in ('raw','ema'):
  data=copy.deepcopy(agent);feature=copy.deepcopy(features);torch.manual_seed(8817);before=calls[0]
  with torch.no_grad(),(decoder.ema.average_parameters(flow.parameters()) if mode=='ema' else nullcontext()):
   assert parameter_digest(flow.named_parameters())==expected[mode]
   feature['pt_token']=flow.model.lane_embed(lane_input).detach();outputs[mode]=flow.sample(data,feature,steps=12)
  assert calls[0]-before==24 and torch.isfinite(outputs[mode]).all();sources[mode]=[data[k].clone() for k in ('gen_noise','_init_diffusion_type_source')];rng[mode]=torch.get_rng_state().clone()
  assert parameter_digest(flow.named_parameters())==expected['raw']
 hook.remove();assert all(torch.equal(a,b) for a,b in zip(sources['raw'],sources['ema']));assert torch.equal(rng['raw'],rng['ema']);assert not torch.equal(outputs['raw'],outputs['ema']);assert not torch.cuda.is_initialized()
 files=('tools/checkpoint_weight_audit_20261015.py','tools/eval_full_checkpoint_midpoint_20261015.py','tools/prove_raw_ema_checkpoint_mode_20261015.py')
 result=dict(status='GENUINE104_RAW_EMA_CPU_SELECTION_AND_NATIVE_SAMPLE_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=json.loads((snapshot/'provenance.json').read_text())['sha256'],step=100256,parameter_digests=expected,actual_decoder_forward_context=context_results,raw_restored_afterEMA=True,both_native_sample_NFE24=True,both_outputs_finite=True,source_noise_and_type_exact=True,post_sample_CPU_RNG_exact=True,outputs_distinct=True,no_CUDA_initialized=True,source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},quality_unproven=True,limitations='CPUcontextandfixtureNNsampleproof; actualGPUrunnerparameterhashaudit remains required before fullqualityclaims. No weights, objective, sampler or metricsmodified.')
 path=BASE/'raw_ema_mode_cpu_proof_20261015.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);os._exit(0)
if __name__=='__main__':main()
