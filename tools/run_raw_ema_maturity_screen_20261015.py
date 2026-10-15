"""Paired diagnostic of96/128epoch raw versusEMA; no training or promotion."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_omega025_e80_confirmation_20261015 as shared
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
ART=BASE/'raw_ema_maturity_screen_v1';RUNS=ROOT/'logs/init_diffusion_continuous_20261014/raw_ema_maturity_screen_v1'
PROFILES={'e96':('omega025_e96_20261015',92544),'final128':('omega025_final128_20261015',123392)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
 files=[Path(__file__),ROOT/'tools/eval_full_checkpoint_midpoint_20261015.py',ROOT/'tools/checkpoint_weight_audit_20261015.py',ROOT/'tools/runtime_cached_torch_loader_sweep.py',ROOT/'tools/torch_pickle_class_cache.py',ROOT/'tools/shared_heading_full/model.py',ROOT/'tools/shared_heading_velocity_20261015.py',*sorted((ROOT/'src/smart').rglob('*.py'))]
 for folder,_ in PROFILES.values():
  files.extend(BASE/'full_checkpoint_snapshots'/folder/name for name in ('config.yaml','provenance.json'))
 return {str(p.relative_to(ROOT)):sha(p) for p in files}
def main():
 assert '.codex_runs' in ROOT.parts,'Run froma frozen source workspace'
 ART.mkdir(exist_ok=False);RUNS.mkdir(exist_ok=False)
 proof=json.loads((BASE/'raw_ema_mode_cpu_proof_20261015.json').read_text());assert proof['status']=='GENUINE104_RAW_EMA_CPU_SELECTION_AND_NATIVE_SAMPLE_PASS'
 for name,digest in proof['source_hashes'].items():assert sha(ROOT/name)==digest
 from tools.checkpoint_weight_audit_20261015 import checkpoint_parameter_digest
 expected={};checkpoints={}
 for label,(folder,step) in PROFILES.items():
  snap=BASE/'full_checkpoint_snapshots'/folder;meta=json.loads((snap/'provenance.json').read_text());assert meta['global_step']==meta['ema_updates']==step and sha(snap/'model.ckpt')==meta['sha256'];checkpoints[label]=meta['sha256'];expected[label]={mode:checkpoint_parameter_digest(snap/'model.ckpt',mode) for mode in ('raw','ema')};assert expected[label]['raw']!=expected[label]['ema']
 frozen=hashes();source_manifest=BASE/'omega025_e104_grid_screen_v2/scenes4096.json';names=json.loads(source_manifest.read_text());assert len(names)==len(set(names))==4096;manifest=ART/'scenes4096.json';shared.save(manifest,names)
 shared.save(ART/'protocol.json',dict(registered_utc=shared.now(),source_snapshot=str(ROOT),source_hashes=frozen,checkpoint_sha256=checkpoints,expected_parameter_digests=expected,GEN=[2817,3817],functional_only_GEN=8817,scenes=4096,manifest_sha256=sha(manifest),weights=['ema','raw'],NFE=24,batch_size=128,precision='medium32true',wait_for_pid=538886,predecessor='native_rectified_speed_prefix8_v3',hypothesis='Separate rawcheckpoint distributiondrift fromEMAaveraging inlate speed/width regression. ExistingEMA96to128full50K speed/width rises inbothGEN.',no_automatic_weight_or_checkpoint_replacement=True,no_full50K_claim=True,no_training_changes=True,no_automatic_failure_retries=True))
 while shared.alive(538886):
  shared.save(ART/'status.json',dict(status='WAITING_FOR_LIVE_RECTIFIED_SPEED_NATIVE8',pid=538886,updated_utc=shared.now()));time.sleep(30)
 assert json.loads((BASE/'native_rectified_speed_prefix8_v3/status.json').read_text())['status']=='NATIVE_RECTIFIED_SPEED_PREFIX8_COMPLETE'
 sampler=ART/'sampler.json';shared.save(sampler,dict(sampling_steps=12,sampling_time_grid=dict(kind='power',power=1.2,terminal_time=.1)))
 tasks=[('e96',mode,256,8817) for mode in ('ema','raw')]+[(label,mode,4096,seed) for label in PROFILES for mode in ('ema','raw') for seed in (2817,3817)]
 rows=[];source={};GT=None
 for label,mode,count,seed in tasks:
  assert hashes()==frozen
  while subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(30)
  selected=manifest if count==4096 else ART/'functional256.json'
  if count!=4096 and not selected.exists():shared.save(selected,json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text())[35000:35256])
  snap=BASE/'full_checkpoint_snapshots'/PROFILES[label][0];tag=f'{label}_{mode}_N{count}_GEN{seed}';output=RUNS/tag;log=RUNS/(tag+'.log')
  command=['taskset','-c','16-31',sys.executable,'-u',str(ROOT/'tools/eval_full_checkpoint_midpoint_20261015.py'),'--checkpoint',str(snap/'model.ckpt'),'--source-config',str(snap/'config.yaml'),'--sampler',str(sampler),'--manifest',str(selected),'--output',str(output),'--integration','native','--batch-size','128','--eval-matmul-precision','medium','--seed',str(seed),'--weights',mode]
  with log.open('x') as f:
   child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
   shared.save(ART/'status.json',dict(status='RUNNING',pid=child.pid,label=tag,log=str(log),updated_utc=shared.now()));result=child.wait()
  assert result==0,f'{tag}failed{result}; preservewithoutretry';assert hashes()==frozen
  r=json.loads((output/'verified.json').read_text());assert r['checkpoint_sha256']==checkpoints[label] and r['checkpoint_step']==PROFILES[label][1];assert r['weights_mode']==mode and r['observed_evaluated_parameter_sha256']==r['expected_evaluated_parameter_sha256']==expected[label][mode];assert set(r['observed_denoiser_calls_per_sample_batch'])=={24}
  if count==4096:
   report=json.loads((output/'sd_agent_metrics.json').read_text());assert report['num_samples']==report['num_gt_samples']==4096 and not report['full_membership']
   h=json.loads((output/'feature_histograms.json').read_text());current={k:v['matched_gt'] for k,v in h.items()}
   if GT is None:GT=current
   assert GT==current
   if seed not in source:source[seed]=r['sampling_source_sha256_per_batch']
   assert source[seed]==r['sampling_source_sha256_per_batch']
  r['profile']=label;rows.append(r);shared.save(ART/'observations.json',rows)
 metrics=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd');comparisons=[];maturity=[]
 for label in PROFILES:
  for seed in (2817,3817):
   pair={r['weights_mode']:r for r in rows if r['profile']==label and r['scenes']==4096 and r['seed']==seed}
   comparisons.append(dict(profile=label,GEN=seed,raw_toEMA={k:pair['raw']['metrics'][k]/pair['ema']['metrics'][k] for k in metrics}))
 for mode in ('ema','raw'):
  for seed in (2817,3817):
   pair={r['profile']:r for r in rows if r['weights_mode']==mode and r['scenes']==4096 and r['seed']==seed}
   maturity.append(dict(weights=mode,GEN=seed,final128_to96={k:pair['final128']['metrics'][k]/pair['e96']['metrics'][k] for k in metrics}))
 shared.save(ART/'decision.json',dict(completed_utc=shared.now(),comparisons=comparisons,maturity=maturity,all_initial_sources_exact=True,all_GT_histograms_exact=True,all_actual_weight_digests_verified=True,no_full50K_quality_claim=True,no_automatic_weight_mode_replacement=True))
 shared.save(ART/'status.json',dict(status='RAW_EMA_MATURITY_SCREEN_COMPLETE',completed_utc=shared.now()))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if ART.exists():shared.save(ART/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(exc),updated_utc=shared.now()))
  raise
