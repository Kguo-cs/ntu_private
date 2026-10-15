"""Audit four fixed-GEN angle histograms without fitting or changing metrics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
from scipy.special import xlogy
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
def read(p):return json.loads(p.read_text())
def main():
 target=read(ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json');scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')
 observations=[]
 for family in ('omega025_e104_confirmation_v1','omega025_e104_holdout_confirmation_v1'):
  observations.extend(r for r in read(BASE/family/'observations.json') if r['scenes']==50000)
 assert sorted(r['seed'] for r in observations)==[2817,3817,5817,6817]
 data={};truth=None;checkpoint=None
 for row in observations:
  output=Path(row['output']);report=read(output/'sd_agent_metrics.json');assert all(report[k]==target[k] for k in scope)
  if checkpoint is None:checkpoint=row['checkpoint_sha256']
  assert checkpoint==row['checkpoint_sha256']
  hp=output/'feature_histograms.json';h=read(hp)['ang_dev_jsd'];g=np.asarray(h['generated'],dtype=float);gt=np.asarray(h['matched_gt'],dtype=float);edges=np.asarray(h['edges'],dtype=float)
  if truth is None:truth=gt
  assert np.array_equal(truth,gt)
  p=g/g.sum();q=gt/gt.sum();mix=(p+q)/2
  c=.5*(xlogy(p,np.divide(p,mix,out=np.ones_like(p),where=mix>0))+xlogy(q,np.divide(q,mix,out=np.ones_like(q),where=mix>0)))*h['multiplier']
  assert math.isclose(c.sum(),report['agent_metrics']['ang_dev_jsd'],rel_tol=1e-9,abs_tol=1e-10)
  centers=(edges[:-1]+edges[1:])/2
  data[row['seed']]=dict(p=p,q=q,c=c,edges=edges,total_generated=g.sum(),feature_count=report['feature_counts_generated'][1],onroad_fraction=report['feature_counts_generated'][1]/report['num_generated_vehicles'],generated_vehicles=report['num_generated_vehicles'],histogram_sha256=hashlib.sha256(hp.read_bytes()).hexdigest(),report=str(output/'sd_agent_metrics.json'),regions={str(cut):dict(generated_mass=float(p[np.abs(centers)>=cut].sum()),GT_mass=float(q[np.abs(centers)>=cut].sum()),JSD_contribution=float(c[np.abs(centers)>=cut].sum())) for cut in (10,30,60,90)})
 reference=(data[2817]['c']+data[3817]['c'])/2;records=[]
 for seed,d in data.items():
  delta=d['c']-reference;ids=np.argsort(delta)[-10:][::-1];edges=d['edges']
  records.append(dict(GEN=seed,JSD=float(d['c'].sum()),passes_absolute_angle=bool(d['c'].sum()<target['targets']['ang_dev_jsd']),delta_vs_original_pair_mean=float(delta.sum()),generated_histogram_count=int(d['total_generated']),onroad_angle_feature_count=d['feature_count'],onroad_fraction=d['onroad_fraction'],generated_vehicles=d['generated_vehicles'],histogram_sha256=d['histogram_sha256'],report=d['report'],absolute_angle_tail_regions=d['regions'],largest_increases_vs_original_pair=[dict(lower=float(edges[i]),upper=float(edges[i+1]),generated_mass=float(d['p'][i]),GT_mass=float(d['q'][i]),JSD_contribution=float(d['c'][i]),increase=float(delta[i])) for i in ids]))
 result=dict(status='FOUR_GEN_SAME104_FULL50K_ANGLE_BIN_AUDIT_COMPLETE',reviewed_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=checkpoint,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_pair=[2817,3817],additional_GEN=[5817,6817],all_reports_scope_verified=True,all_JSD_recomputed_exactly=True,records=records,limitations='Offline aggregate histograms localize sampling variation, not causal per-agent/map attribution. Region cutoffs are descriptive only; no metric, checkpoint, source, postprocessing or acceptance changes.')
 path=BASE/'omega104_four_GEN_angle_bin_audit_20261015.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
