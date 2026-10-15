"""Read-only metric review against the user's absolute SD/VW target vector."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
SCOPE=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256',
       'official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')

def read(p):return json.loads(Path(p).read_text())
def main():
    target=read(ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json');reference=read(ROOT/'configs/metrics/iu3fa4gx_full50000.json')
    rows=[];seen=set()
    for family in ('full_sampler_50k_v2','deep_sampler_v1','deep_native_seed0_v1','native_midpoint_seed0_v1','deep_final128_midpoint_v1','omega025_e80_confirmation_v1','omega025_e96_confirmation_v1','omega025_e104_confirmation_v1','omega025_final128_confirmation_v1','shape4_e64_sampler_review_v2'):
        path=BASE/family/'observations.json'
        if not path.exists():continue
        for row in read(path):
            if row.get('scenes')!=50000:continue
            output=Path(row['output']);report_path=output/'sd_agent_metrics.json'
            if str(report_path) in seen:continue
            seen.add(str(report_path));report=read(report_path)
            assert all(report.get(k)==target[k] for k in SCOPE),str(report_path)
            m=report['agent_metrics'];ratios={k:m[k]/v for k,v in target['targets'].items()}
            passed=[k for k,v in ratios.items() if v<1]
            pair_id=dict(checkpoint=row.get('checkpoint_sha256'),sampler=row.get('sampler'),integration=row.get('integration'),precision=row.get('matmul_precision'),
                         effective_integrator=row.get('effective_integrator',row.get('integration')),
                         sampling_batch_count=len(row.get('sampling_source_sha256_per_batch',[])),
                         source=hashlib.sha256(json.dumps(row.get('smart_source_sha256'),sort_keys=True).encode()).hexdigest())
            rows.append(dict(label=row.get('snapshot',row.get('model',output.name)),family=family,GEN=row.get('seed'),checkpoint_step=row.get('checkpoint_step'),
                full128epoch_training=row.get('checkpoint_step')==123392,report=str(report_path),metrics=m,
                target_ratios=ratios,passed_targets=passed,passed_count=len(passed),worst_target_ratio=max(ratios.values()),
                binding_metric=max(ratios,key=ratios.get),R_focus_target=math.fsum(ratios[k] for k in ('speed_jsd','ang_dev_jsd','length_jsd'))/3,
                R_jsd_657=math.fsum(m[k]/reference['agent_metrics'][k] for k in ratios if k!='collision_rate')/6,
                pair_key=hashlib.sha256(json.dumps(pair_id,sort_keys=True).encode()).hexdigest()))
    # These native reports have different batch/noise streams from the archived
    # batch128 pairs. Keep their independent evidence without merging pair IDs.
    extras=[]
    omega80=BASE/'full_checkpoint_snapshots/omega025_e80_20261015/native_report.json'
    if omega80.exists():
        extras.append(dict(report=str(omega80),label='omega025_native_e80_preserved_snapshot',
                           family='native_e80_single_validation',GEN=None))
    seed0=BASE/'deep_final128_native_seed0_review_20261015.json'
    if seed0.exists():
        r=read(seed0)
        extras.append(dict(report=r['report'],label='deep_final128_native_seed0_batch64',
                           family='native_seed0_complete50K',GEN=0))
    for family in ('deep_final_timegrid_v1', 'deep_final_timegrid_full_controls_v1'):
        grid=BASE/family/'observations.json'
        if grid.exists():
            for r in read(grid):
                if r['scenes']==50000:
                    extras.append(dict(report=r['report'],label=f"deep_final128_p{r['power']}_batch64",
                        family=family,GEN=r['seed'],
                        pair_key=f"{family}_batch64_power{r['power']}"))
    for extra in extras:
        report=read(extra['report']);assert all(report.get(k)==target[k] for k in SCOPE)
        m=report['agent_metrics'];ratios={k:m[k]/v for k,v in target['targets'].items()}
        step=report['decoder']['ema_num_updates']
        rows.append(dict(**extra,checkpoint_step=step,full128epoch_training=step==123392,
            metrics=m,target_ratios=ratios,passed_targets=[k for k,v in ratios.items() if v<1],
            passed_count=sum(v<1 for v in ratios.values()),worst_target_ratio=max(ratios.values()),
            binding_metric=max(ratios,key=ratios.get),
            R_focus_target=math.fsum(ratios[k] for k in ('speed_jsd','ang_dev_jsd','length_jsd'))/3,
            R_jsd_657=m['R_jsd']))
    fleet=BASE/'latest_fleet_full_validation_20261015.json'
    native=read(fleet if fleet.exists() else BASE/'latest_full_validation_review_20261015.json')
    for record in native['reports']:
        if 'report' not in record:continue
        report=record['report'];assert all(report.get(k)==target[k] for k in SCOPE)
        m=report['agent_metrics'];ratios={k:m[k]/v for k,v in target['targets'].items()}
        rows.append(dict(label=record['experiment'],family='native_full_training_validation',GEN=None,checkpoint_step=report['decoder']['ema_num_updates'],
            full128epoch_training=report['decoder']['ema_num_updates']==123392,report=record['path'],metrics=m,target_ratios=ratios,
            passed_targets=[k for k,v in ratios.items() if v<1],passed_count=sum(v<1 for v in ratios.values()),worst_target_ratio=max(ratios.values()),
            binding_metric=max(ratios,key=ratios.get),R_focus_target=math.fsum(ratios[k] for k in ('speed_jsd','ang_dev_jsd','length_jsd'))/3,
            R_jsd_657=m.get('R_jsd')))
    groups={}
    for row in rows:
        if 'pair_key' in row:groups.setdefault(row['pair_key'],[]).append(row)
    robust=[]
    for key,group in groups.items():
        selected={r['GEN']:r for r in group if r['GEN'] in (2817,3817)}
        if len(selected)!=2:continue
        robust.append(dict(pair_key=key,labels=[r['label'] for r in selected.values()],both_GEN_all7_pass=all(r['passed_count']==7 for r in selected.values()),
            worst_target_ratio=max(r['worst_target_ratio'] for r in selected.values()),rows=[dict(GEN=r['GEN'],passed_count=r['passed_count'],report=r['report']) for r in selected.values()]))
    record=dict(reviewed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),target=target['targets'],
        selection_note='For already fully trained checkpoints, absolute target violations drive the final decision. R_jsd against657 remains descriptive; a relative geometry regression is not itself an SD/VW target failure.',
        conditioning_caveat='Our fixedGTego pose/shape/type and SMART map representation differ from releasedSD/VW; this is the agreed repository50K metric comparison, not identical paper conditioning.',
        rows=sorted(rows,key=lambda r:(-r['passed_count'],r['worst_target_ratio'],r['R_focus_target'])),robust_pairs=robust,
        both_GEN_same_checkpoint_absolute_metric_gate_pass=any(g['both_GEN_all7_pass'] for g in robust))
    (BASE/'current_target_frontier_20261015.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),robust_pairs=len(robust),both_GEN_all7_pass=record['both_GEN_same_checkpoint_absolute_metric_gate_pass'],top=[{k:r[k] for k in ('label','GEN','passed_count','binding_metric','worst_target_ratio')} for r in record['rows'][:5]])))

if __name__=='__main__':main()
