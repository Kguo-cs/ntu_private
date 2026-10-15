"""Two additional fixed GENs for genuine final128 weights, not new training."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_omega025_e80_confirmation_20261015 as shared
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014';ART=BASE/'omega025_final128_additional_GEN_v1';RUNS=ROOT/'logs/init_diffusion_continuous_20261014/omega025_final128_additional_GEN_v1';SNAP=BASE/'full_checkpoint_snapshots/omega025_final128_20261015'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
 shared.SNAPSHOT=SNAP
 return dict(shared.hashes(),**{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'tools/checkpoint_weight_audit_20261015.py')})
def main():
 assert '.codex_runs' in ROOT.parts
 ART.mkdir(exist_ok=False);RUNS.mkdir(exist_ok=False)
 meta=json.loads((SNAP/'provenance.json').read_text());assert meta['epoch']==127 and meta['global_step']==meta['ema_updates']==123392 and meta['sha256']==sha(SNAP/'model.ckpt')=='570102b2a878f1a74db7a3f8c550819248010b49483d6f8bb5d416848580aa45'
 screened=[r for r in json.loads((BASE/'raw_ema_maturity_screen_v1/observations.json').read_text()) if r['profile']=='final128' and r['weights_mode']=='ema'];assert len(screened)==2
 expected=screened[0]['observed_evaluated_parameter_sha256'];assert all(r['observed_evaluated_parameter_sha256']==expected and r['checkpoint_sha256']==meta['sha256'] for r in screened)
 assert not any(r.get('seed') in (5817,6817) for r in json.loads((BASE/'omega025_final128_confirmation_v1/observations.json').read_text()))
 frozen=hashes();shared.save(ART/'protocol.json',dict(registered_utc=shared.now(),source_snapshot=str(ROOT),source_hashes=frozen,checkpoint_sha256=meta['sha256'],checkpoint_epoch=127,checkpoint_step=123392,expected_EMA_parameter_sha256=expected,GEN=[5817,6817],previous_GEN=[2817,3817],functional_only_GEN=8817,scenes=50000,NFE=24,batch_size=128,precision='medium32true',sampler=dict(integrator='native_midpoint',power=1.2,terminal_time=.1),wait_for_pid=547242,predecessor='omega05_e64_confirmation_v1',reason='Originalfinal128GEN2817/3817all7pass. Selected104hasaGEN5817anglefailure; testrobustnessofunchangedfinal128weightsbeforeanyreplacement. TheseGENsarepreviouslyusedatothercheckpoints, notgloballyunseen.',no_automatic_winner_replacement=True,no_automatic_failure_retries=True,no_merging_model_or_metrics=True,limitation='OneTRAINseed, agreedrepositoryfixedGTego/SMARTmapconditiondiffersfromreleasedSD/VW. FourGENtestisadditionalrobustnessnotredefinitionoforiginalnumericmilestone.'))
 while shared.alive(547242):
  shared.save(ART/'status.json',dict(status='WAITING_FOR_LIVE_OMEGA05_E64_CONFIRMATION',pid=547242,updated_utc=shared.now()));time.sleep(30)
 assert json.loads((BASE/'omega05_e64_confirmation_v1/status.json').read_text())['status']=='OMEGA05_E64_CONFIRMATION_COMPLETE'
 names=json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text());assert len(names)==len(set(names))==50000;target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text());sampler=ART/'sampler.json';shared.save(sampler,dict(sampling_steps=12,sampling_time_grid=dict(kind='power',power=1.2,terminal_time=.1)));rows=[]
 for count,seed in [(256,8817),(50000,5817),(50000,6817)]:
  assert hashes()==frozen
  while subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
  tag=f'omega025_final128_additional_ema_MP24_N{count}_GEN{seed}';manifest=ART/(tag+'.manifest.json');shared.save(manifest,names if count==50000 else names[35000:35256]);output=RUNS/tag;log=RUNS/(tag+'.log')
  command=['taskset','-c','16-31',sys.executable,'-u',str(shared.RUNNER),'--checkpoint',str(SNAP/'model.ckpt'),'--source-config',str(SNAP/'config.yaml'),'--sampler',str(sampler),'--manifest',str(manifest),'--output',str(output),'--integration','native','--batch-size','128','--eval-matmul-precision','medium','--seed',str(seed),'--weights','ema']
  if count==50000:command+=['--full-set','--r-reference',str(ROOT/'configs/metrics/iu3fa4gx_full50000.json')]
  with log.open('x') as f:
   child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
   shared.save(ART/'status.json',dict(status='RUNNING',pid=child.pid,label=tag,log=str(log),updated_utc=shared.now()));result=child.wait()
  assert result==0,f'{tag}failed{result}; preservewithoutretry';assert hashes()==frozen
  row=json.loads((output/'verified.json').read_text());report=json.loads((output/'sd_agent_metrics.json').read_text());assert row['checkpoint_sha256']==meta['sha256'] and row['checkpoint_step']==report['decoder']['ema_num_updates']==123392;assert row['observed_evaluated_parameter_sha256']==expected and row['weights_mode']=='ema';assert set(row['observed_denoiser_calls_per_sample_batch'])=={24}
  if count==50000:
   for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):assert report[key]==target[key]
   row['target_ratios']={k:row['metrics'][k]/v for k,v in target['targets'].items()};row['passed_targets']=[k for k,v in row['target_ratios'].items() if v<1]
  rows.append(row);shared.save(ART/'observations.json',rows)
 allrows=[r for r in json.loads((BASE/'omega025_final128_confirmation_v1/observations.json').read_text()) if r['scenes']==50000]+rows[1:];assert {r['seed'] for r in allrows}=={2817,3817,5817,6817}
 shared.save(ART/'decision.json',dict(completed_utc=shared.now(),all4GEN_all7=all(all(r['metrics'][k]<v for k,v in target['targets'].items()) for r in allrows),results=[dict(GEN=r['seed'],metrics=r['metrics']) for r in allrows],no_automatic_winner_replacement=True))
 shared.save(ART/'status.json',dict(status='OMEGA025_FINAL128_ADDITIONAL_GEN_COMPLETE',completed_utc=shared.now()))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if ART.exists():shared.save(ART/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(exc),updated_utc=shared.now()))
  raise
