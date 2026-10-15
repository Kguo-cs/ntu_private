"""Confirm the preregistered omega0.05 epoch64 checkpoint on both fixed GENs."""
from pathlib import Path
import gc,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_omega025_e80_confirmation_20261015 as shared
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014';ART=BASE/'omega05_e64_confirmation_v1';RUNS=ROOT/'logs/init_diffusion_continuous_20261014/omega05_e64_confirmation_v1';SNAP=BASE/'full_checkpoint_snapshots/omega05_e64_20261015'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
 shared.SNAPSHOT=SNAP
 return dict(shared.hashes(include_snapshot=False),**{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'tools/checkpoint_weight_audit_20261015.py')})
def cpu_preflight():
 import jinja2,torch,hydra
 from omegaconf import OmegaConf,open_dict
 from tools.checkpoint_weight_audit_20261015 import parameter_digest,checkpoint_parameter_digest
 gc.disable();torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
 cfg=OmegaConf.load(SNAP/'config.yaml')
 with open_dict(cfg):cfg.model.model_config.decoder.init_diffusion.normalizer_path=None;cfg.paths.output_dir='/tmp/sim_omega05_e64_cpu'
 model=hydra.utils.instantiate(cfg.model,_recursive_=False);checkpoint=torch.load(SNAP/'model.ckpt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();decoder=model.encoder.init_decoder
 assert checkpoint['epoch']==63 and checkpoint['global_step']==decoder.ema.num_updates==61696
 assert decoder.component_loss['components']['heading']['weight']==.05 and decoder.denoiser_options['heading_velocity_readout']=='shared_state_row3'
 assert sum(p.numel() for p in decoder.G1.parameters())==8083978
 assert parameter_digest(model.encoder.map_encoder.named_parameters())=='b2b95bdfd833032f2bbaaacf61df8c9d117edb23a877090016f2fa45d1d79be8'
 assert not torch.cuda.is_initialized()
 return dict(status='GENUINE_OMEGA05_E64_CPU_STRICT_LOAD_PASS',checkpoint_sha256=sha(SNAP/'model.ckpt'),step=61696,parameters=8083978,heading_weight=.05,frozen_map_weights_exact=True,expected_EMA_parameter_sha256=checkpoint_parameter_digest(SNAP/'model.ckpt','ema'),no_CUDA=True)
def main():
 assert '.codex_runs' in ROOT.parts,'Use frozen source workspace'
 ART.mkdir(exist_ok=False);RUNS.mkdir(exist_ok=False)
 registration=BASE/'omega05_e64_registration_20261015.json';prior=json.loads(registration.read_text());assert prior['EMA']==61696 and prior['GEN']==[2817,3817]
 frozen=hashes();shared.save(ART/'protocol.json',dict(registered_utc=shared.now(),prior_registration=str(registration),prior_registration_sha256=sha(registration),source_snapshot=str(ROOT),source_hashes=frozen,checkpoint_epoch=63,checkpoint_step=61696,heading_weight=.05,GEN=[2817,3817],functional_only_GEN=8817,scenes=50000,NFE=24,batch_size=128,precision='medium32true',sampling=dict(integrator='native_midpoint',power=1.2,terminal_time=.1),wait_for_preservation_pid=544531,wait_for_GPU_predecessor_pid=546274,selection='Fixed64epoch andGENs registered before its native/GENquality. Preserveandreview allresults, no automaticcheckpointreplacement.',no_automatic_failure_retries=True,limitation='OneTRAINseed. FixedGTego andSMARTroadmapconditioning differfromSD/VW release; agreedrepository50Kcomparison.'))
 while shared.alive(544531):
  shared.save(ART/'status.json',dict(status='WAITING_FOR_LIVE_REGISTERED64_PRESERVATION',pid=544531,updated_utc=shared.now()));time.sleep(30)
 preservation=json.loads((BASE/'omega05_e64_preservation_v1/status.json').read_text());assert preservation['status']=='OMEGA05_E64_GENUINE_CHECKPOINT_PRESERVED','Reviewpreservationfailurebeforenextstage'
 assert hashes()==frozen;meta=json.loads((SNAP/'provenance.json').read_text());assert meta['epoch']==63 and meta['global_step']==meta['ema_updates']==61696 and sha(SNAP/'model.ckpt')==meta['sha256']
 shared.prepare_portable_config(SNAP,ART);proof=cpu_preflight();shared.save(ART/'cpu_preflight.json',proof);config_sha=sha(SNAP/'config.yaml')
 while shared.alive(546274):
  shared.save(ART/'status.json',dict(status='WAITING_FOR_LIVE_RAW_EMA_SCREEN',pid=546274,checkpoint_sha256=proof['checkpoint_sha256'],updated_utc=shared.now()));time.sleep(30)
 assert json.loads((BASE/'raw_ema_maturity_screen_v1/status.json').read_text())['status']=='RAW_EMA_MATURITY_SCREEN_COMPLETE'
 names=json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text());assert len(names)==len(set(names))==50000;target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text());sampler=ART/'sampler.json';shared.save(sampler,dict(sampling_steps=12,sampling_time_grid=dict(kind='power',power=1.2,terminal_time=.1)));rows=[]
 for count,seed in [(256,8817),(50000,2817),(50000,3817)]:
  assert hashes()==frozen and sha(SNAP/'config.yaml')==config_sha
  while subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
  label=f'omega05_e64_ema_nativeMP24_N{count}_GEN{seed}';manifest=ART/(label+'.manifest.json');shared.save(manifest,names if count==50000 else names[35000:35256]);output=RUNS/label;log=RUNS/(label+'.log')
  command=['taskset','-c','16-31',sys.executable,'-u',str(shared.RUNNER),'--checkpoint',str(SNAP/'model.ckpt'),'--source-config',str(SNAP/'config.yaml'),'--sampler',str(sampler),'--manifest',str(manifest),'--output',str(output),'--integration','native','--batch-size','128','--eval-matmul-precision','medium','--seed',str(seed),'--weights','ema']
  if count==50000:command+=['--full-set','--r-reference',str(ROOT/'configs/metrics/iu3fa4gx_full50000.json')]
  with log.open('x') as f:
   child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
   shared.save(ART/'status.json',dict(status='RUNNING',pid=child.pid,label=label,log=str(log),updated_utc=shared.now()));result=child.wait()
  assert result==0,f'{label}failed{result}; preservewithoutretry';assert hashes()==frozen
  row=json.loads((output/'verified.json').read_text());report=json.loads((output/'sd_agent_metrics.json').read_text());assert row['checkpoint_step']==report['decoder']['ema_num_updates']==61696 and row['checkpoint_sha256']==proof['checkpoint_sha256'];assert set(row['observed_denoiser_calls_per_sample_batch'])=={24};assert row['weights_mode']=='ema' and row['observed_evaluated_parameter_sha256']==proof['expected_EMA_parameter_sha256']
  if count==50000:
   for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):assert report[key]==target[key]
   row['target_ratios']={k:row['metrics'][k]/v for k,v in target['targets'].items()};row['passed_targets']=[k for k,v in row['target_ratios'].items() if v<1]
  rows.append(row);shared.save(ART/'observations.json',rows)
 control={r['seed']:r for r in json.loads((BASE/'omega025_e104_confirmation_v1/observations.json').read_text()) if r['scenes']==50000};comparisons=[dict(GEN=r['seed'],initial_source_exact=r['sampling_source_sha256_per_batch']==control[r['seed']]['sampling_source_sha256_per_batch'],ratios_to104={k:r['metrics'][k]/control[r['seed']]['metrics'][k] for k in target['targets']}) for r in rows[1:]]
 shared.save(ART/'decision.json',dict(completed_utc=shared.now(),both_GEN_all7=all(len(r['passed_targets'])==7 for r in rows[1:]),results=[dict(GEN=r['seed'],metrics=r['metrics'],passed_targets=r['passed_targets']) for r in rows[1:]],comparisons_to104=comparisons,no_automatic_winner_replacement=True,actual64epoch_not_final128=True))
 shared.save(ART/'status.json',dict(status='OMEGA05_E64_CONFIRMATION_COMPLETE',completed_utc=shared.now()))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if ART.exists():shared.save(ART/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(exc),updated_utc=shared.now()))
  raise
