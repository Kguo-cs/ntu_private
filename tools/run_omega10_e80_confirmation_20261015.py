"""Paired full50K review of the preselected native80epoch omega0.1 weights."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_omega025_e80_confirmation_20261015 as shared
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
ART=BASE/'omega10_e80_confirmation_v1'
RUNS=ROOT/'logs/init_diffusion_continuous_20261014/omega10_e80_confirmation_v1'
SNAPSHOT=BASE/'full_checkpoint_snapshots/omega10_e80_20261015'

def hashes():
 shared.SNAPSHOT=SNAPSHOT
 return dict(shared.hashes(),**{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

def main():
 ART.mkdir(exist_ok=False);RUNS.mkdir(exist_ok=False)
 proof=json.loads((BASE/'omega10_e80_cpu_preflight_20261015.json').read_text())
 assert proof['status']=='OMEGA10_E80_REAL_CHECKPOINT_CPU_STRICT_LOAD_PASS' and proof['step']==77120
 assert hashlib.sha256((SNAPSHOT/'model.ckpt').read_bytes()).hexdigest()==proof['checkpoint_sha256']
 assert hashlib.sha256((SNAPSHOT/'config.yaml').read_bytes()).hexdigest()==proof['portable_config_sha256']
 frozen=hashes()
 protocol=dict(registered_utc=shared.now(),source_hashes=frozen,checkpoint_sha256=proof['checkpoint_sha256'],checkpoint_epoch=79,
  checkpoint_step=77120,heading_weight=.1,GEN=[2817,3817],functional_only_GEN=8817,batch_size=128,
  matmul_precision='medium',trainer_precision='32-true',NFE=24,integrator='native_midpoint',power=1.2,terminal_time=.1,
  selection='Native80epoch full50K firstpassesall7; weightssaved andseedsregistered beforeobservingindependentGENquality. Nocherrypickingseeds orcheckpointmerging.',
  wait_for_pid=532726,wait_for_study='omega025_e104_holdout_confirmation_v1',
  no_automatic_failure_retries=True,no_automatic_winner_replacement=True,
  limitation='OneTRAINseed. Traininghardware differsfromomega025; notastrict headingweightablation. GTegopose/shape/type fixed andSMARTroadedges differfromSD/VW releasedconditions.')
 shared.save(ART/'protocol.json',protocol)
 while shared.alive(532726):
  shared.save(ART/'status.json',dict(status='WAITING_FOR_LIVE_HELDOUT104_STUDY',pid=532726,updated_utc=shared.now()));time.sleep(30)
 predecessor=json.loads((BASE/'omega025_e104_holdout_confirmation_v1/status.json').read_text())
 assert predecessor['status']=='OMEGA025_E104_HOLDOUT_CONFIRMATION_COMPLETE'
 names=json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text());assert len(names)==len(set(names))==50000
 target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
 sampler=ART/'sampler.json';shared.save(sampler,dict(sampling_steps=12,sampling_time_grid=dict(kind='power',power=1.2,terminal_time=.1)))
 rows=[]
 for count,seed in [(256,8817),(50000,2817),(50000,3817)]:
  assert hashes()==frozen
  while subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
   shared.save(ART/'status.json',dict(status='WAITING_FOR_FREE_LOCAL_GPU',updated_utc=shared.now()));time.sleep(30)
  label=f'omega10_e80_nativeMP24_N{count}_GEN{seed}';manifest=ART/(label+'.manifest.json');shared.save(manifest,names if count==50000 else names[35000:35256]);output=RUNS/label;log=RUNS/(label+'.log')
  command=['taskset','-c','16-31',sys.executable,'-u',str(shared.RUNNER),'--checkpoint',str(SNAPSHOT/'model.ckpt'),'--source-config',str(SNAPSHOT/'config.yaml'),'--sampler',str(sampler),'--manifest',str(manifest),'--output',str(output),'--integration','native','--batch-size','128','--eval-matmul-precision','medium','--seed',str(seed)]
  if count==50000:command+=['--full-set','--r-reference',str(ROOT/'configs/metrics/iu3fa4gx_full50000.json')]
  with log.open('x') as f:
   child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
   shared.save(ART/'status.json',dict(status='RUNNING',pid=child.pid,log=str(log),label=label,updated_utc=shared.now()));result=child.wait()
  assert result==0,f'{label} failed{result}; preserve withoutautomaticretry'
  assert hashes()==frozen
  row=json.loads((output/'verified.json').read_text());report=json.loads((output/'sd_agent_metrics.json').read_text())
  assert row['checkpoint_step']==report['decoder']['ema_num_updates']==77120
  assert row['checkpoint_sha256']==proof['checkpoint_sha256'] and set(row['observed_denoiser_calls_per_sample_batch'])=={24}
  if count==50000:
   for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):assert report[key]==target[key]
   assert report['agent_metrics']['R_scope_matches_reference']==1
   row['target_ratios']={k:row['metrics'][k]/v for k,v in target['targets'].items()};row['passed_targets']=[k for k,v in row['target_ratios'].items() if v<1]
  rows.append(row);shared.save(ART/'observations.json',rows)
 prior={r['seed']:r for r in json.loads((BASE/'omega025_e104_confirmation_v1/observations.json').read_text()) if r['scenes']==50000}
 comparisons=[dict(seed=r['seed'],source_noise_exactly_matched=r['sampling_source_sha256_per_batch']==prior[r['seed']]['sampling_source_sha256_per_batch'],metric_ratios_to104={k:r['metrics'][k]/prior[r['seed']]['metrics'][k] for k in target['targets']}) for r in rows[1:]]
 shared.save(ART/'decision.json',dict(completed_utc=shared.now(),same_checkpoint_two_independent_GEN_full50K=True,both_GEN_all7=all(len(r['passed_targets'])==7 for r in rows[1:]),results=[dict(seed=r['seed'],metrics=r['metrics'],passed_targets=r['passed_targets']) for r in rows[1:]],comparisons_to104=comparisons,no_automatic_winner_replacement=True,final128_weights=False))
 shared.save(ART/'status.json',dict(status='OMEGA10_E80_CONFIRMATION_COMPLETE',completed_utc=shared.now()))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if ART.exists():shared.save(ART/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(exc),updated_utc=shared.now()))
  raise
