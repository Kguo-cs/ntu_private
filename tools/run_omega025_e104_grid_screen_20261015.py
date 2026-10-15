"""NFE24 sampler-only screen, fixed verified104 weights; no model fitting."""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_omega025_e80_confirmation_20261015 as shared
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
SNAPSHOT=BASE/'full_checkpoint_snapshots/omega025_e104_20261015'
ART=BASE/'omega025_e104_grid_screen_v1'
RUNS=ROOT/'logs/init_diffusion_continuous_20261014/omega025_e104_grid_screen_v1'
OPTIONS={'control':(1.2,.1),'power08':(.8,.1),'power16':(1.6,.1),'terminal15':(1.2,.15),'terminal20':(1.2,.2)}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes():
 shared.SNAPSHOT=SNAPSHOT
 return dict(shared.hashes(),**{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})

def cpu_preflight():
 import gc;gc.disable()
 import copy,importlib.util,jinja2,torch,hydra
 from omegaconf import OmegaConf,open_dict
 from src.smart.diffusion.sampling_time_grid import parse_sampling_time_grid
 torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
 cfg=OmegaConf.load(SNAPSHOT/'config.yaml')
 with open_dict(cfg):
  cfg.model.model_config.decoder.init_diffusion.normalizer_path=None
  cfg.paths.output_dir='/tmp/sim_grid_screen_cpu'
 model=hydra.utils.instantiate(cfg.model,_recursive_=False)
 checkpoint=torch.load(SNAPSHOT/'model.ckpt',map_location='cpu',weights_only=False)
 model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
 decoder=model.encoder.init_decoder;decoder.ema.copy_to(decoder.G1.parameters());flow=decoder.G1
 assert checkpoint['global_step']==decoder.ema.num_updates==100256
 spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test_init_diffusion_ego_partial.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 _,agent,feature=fixture.InitDiffusionEgoPartialTest().inputs();decoder._prepare_ego_context(agent)
 torch.manual_seed(32415);feature['pt_token']=flow.model.lane_embed(torch.randn(3,128)).detach()
 calls=[0];hook=flow.model.register_forward_pre_hook(lambda *unused:calls.__setitem__(0,calls[0]+1));records=[];source=None
 for label,(power,terminal) in OPTIONS.items():
  data=copy.deepcopy(agent);flow.sampling_time_grid=parse_sampling_time_grid(dict(kind='power',power=power,terminal_time=terminal))
  torch.manual_seed(8817);before=calls[0]
  with torch.no_grad():result=flow.sample(data,feature,steps=12)
  assert calls[0]-before==24 and torch.isfinite(result).all()
  current=[data[k].detach().clone() for k in ('gen_noise','_init_diffusion_type_source')]
  if source is None:source=current
  else:assert all(torch.equal(a,b) for a,b in zip(source,current))
  records.append(dict(label=label,power=power,terminal_time=terminal,NFE=24,finite=True,same_initial_source=True))
 hook.remove()
 proof=dict(status='FIXED104_NFE24_NATIVE_GRID_CPU_PREFLIGHT_PASS',checkpoint_sha256=sha(SNAPSHOT/'model.ckpt'),EMA=100256,source_hashes=hashes(),settings=records,quality_unproven=True,minimum_supported_terminal=.1)
 shared.save(BASE/'omega025_e104_grid_cpu_preflight_20261015.json',proof);print(json.dumps({k:v for k,v in proof.items() if k!='source_hashes'}),flush=True);os._exit(0)

def main():
 import numpy as np
 ART.mkdir(exist_ok=False);RUNS.mkdir(exist_ok=False)
 proof=json.loads((BASE/'omega025_e104_grid_cpu_preflight_20261015.json').read_text());assert proof['status']=='FIXED104_NFE24_NATIVE_GRID_CPU_PREFLIGHT_PASS'
 assert proof['source_hashes']==hashes()
 names=json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text());assert len(names)==len(set(names))==50000
 ids=sorted(np.random.default_rng(151024).choice(len(names),4096,replace=False).tolist());selected=[names[i] for i in ids];manifest=ART/'scenes4096.json';shared.save(manifest,selected)
 frozen=hashes()
 shared.save(ART/'protocol.json',dict(registered_utc=shared.now(),checkpoint_sha256=proof['checkpoint_sha256'],settings=OPTIONS,GEN=[2817,3817],scenes=4096,selection_seed=151024,manifest_sha256=sha(manifest),
  NFE=24,batch_size=128,precision='medium32true',wait_for_live_pid=533812,source_hashes=frozen,
  hypothesis='Physicalx0attribute bias maydepend onintegrationgrid/endtime. Test supportedterminal>=.1 andinteriordensity; nochange toweights/losses/objective ormetricbins.',
  ranking='Meanpaired ratio ofspeed/angle/length; eligible onlyif focus<=.97, mean6JSD<=1, eachfocusmetric<=1.05 andotherJSD<=1.10 ineachGEN.',
  no_automatic_full50K_promotion=True,no_automatic_failure_retries=True,no_full50K_claim_from4096=True))
 while shared.alive(533812):
  shared.save(ART/'status.json',dict(status='WAITING_FOR_LIVE_OMEGA10_E80_REVIEW',pid=533812,updated_utc=shared.now()));time.sleep(30)
 assert json.loads((BASE/'omega10_e80_confirmation_v1/status.json').read_text())['status']=='OMEGA10_E80_CONFIRMATION_COMPLETE'
 rows=[]
 for label,(power,terminal) in OPTIONS.items():
  sampler=ART/(label+'_sampler.json');shared.save(sampler,dict(sampling_steps=12,sampling_time_grid=dict(kind='power',power=power,terminal_time=terminal)))
  for seed in (2817,3817):
   assert hashes()==frozen
   while subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
   output=RUNS/(label+'_GEN'+str(seed));log=RUNS/(label+'_GEN'+str(seed)+'.log')
   cmd=['taskset','-c','16-31',sys.executable,'-u',str(shared.RUNNER),'--checkpoint',str(SNAPSHOT/'model.ckpt'),'--source-config',str(SNAPSHOT/'config.yaml'),'--sampler',str(sampler),'--manifest',str(manifest),'--output',str(output),'--integration','native','--batch-size','128','--eval-matmul-precision','medium','--seed',str(seed)]
   with log.open('x') as f:
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
    shared.save(ART/'status.json',dict(status='RUNNING',label=label,GEN=seed,pid=child.pid,log=str(log),updated_utc=shared.now()));result=child.wait()
   assert result==0,f'{label}/{seed} failed{result}; noautomaticretry'
   assert hashes()==frozen
   row=json.loads((output/'verified.json').read_text());assert row['checkpoint_step']==100256 and set(row['observed_denoiser_calls_per_sample_batch'])=={24}
   report=json.loads((output/'sd_agent_metrics.json').read_text());assert report['num_samples']==report['num_gt_samples']==4096 and not report['full_membership']
   row['option']=label;rows.append(row);shared.save(ART/'observations.json',rows)
 control={r['seed']:r for r in rows if r['option']=='control'};decisions=[]
 for label in OPTIONS:
  if label=='control':continue
  comparisons=[]
  for seed in (2817,3817):
   row=next(r for r in rows if r['option']==label and r['seed']==seed);old=control[seed]
   assert row['sampling_source_sha256_per_batch']==old['sampling_source_sha256_per_batch']
   ratios={k:row['metrics'][k]/old['metrics'][k] for k in old['metrics'] if k.endswith('_jsd')}
   focus=sum(ratios[k] for k in ('speed_jsd','ang_dev_jsd','length_jsd'))/3
   eligible=focus<=.97 and sum(ratios.values())/6<=1 and all(v<=(1.05 if k in ('speed_jsd','ang_dev_jsd','length_jsd') else 1.10) for k,v in ratios.items())
   comparisons.append(dict(GEN=seed,ratios=ratios,focus_ratio=focus,early_screen_eligible=eligible))
  decisions.append(dict(option=label,mean_focus_ratio=sum(r['focus_ratio'] for r in comparisons)/2,both_GEN_early_gate=all(r['early_screen_eligible'] for r in comparisons),comparisons=comparisons))
 shared.save(ART/'decision.json',dict(completed_utc=shared.now(),candidates=sorted(decisions,key=lambda r:r['mean_focus_ratio']),source_exactly_matched=True,no_full50K_quality_claim=True,no_automatic_full_promotion=True))
 shared.save(ART/'status.json',dict(status='FIXED104_NFE24_GRID_SCREEN_COMPLETE',completed_utc=shared.now()))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 try:cpu_preflight() if args.preflight else main()
 except Exception as exc:
  if ART.exists():shared.save(ART/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(exc),updated_utc=shared.now()))
  raise
