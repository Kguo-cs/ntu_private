"""Independent native histogram/count and provenance checks for size transport."""
from pathlib import Path
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
from tools.audit_init_overnight_20261014 import divergence,METRICS
ART=ROOT/'output/reports/init_diffusion_expanded_20261014/size_calibration'
UNAFFECTED=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','speed_jsd')


def main():
    protocol=read(ART/'protocol.json')
    for path,digest in protocol['sources'].items():assert sha(ROOT/path)==digest,path
    cells=[];comparisons={}
    for filename in ('quick_comparison.json','full50000_comparison.json'):
        path=ART/filename
        if not path.exists():continue
        comparison=read(path);comparisons[filename]=sha(path)
        for row in comparison['rows']:
            baseline=next(r for r in comparison['raw_rows'] if r['eval_seed']==row['eval_seed'])
            assert all(row[k]==baseline[k] for k in ('checkpoint','seed','budget','eval_seed','eval_manifest_sha256','train_order_sha256','initial_map_sha256','evaluation_weights'))
            report=read(row['metrics_path']);raw_report=read(baseline['metrics_path'])
            hist=read(Path(row['metrics_path']).parent/'feature_histograms.json')
            raw_hist=read(Path(baseline['metrics_path']).parent/'feature_histograms.json')
            post=report['postprocessing'];calibration=Path(post['calibration_file'])
            artifact=read(calibration);provenance=artifact['provenance']
            assert post['calibration_sha256']==sha(calibration)
            assert post['checkpoint_sha256']==provenance['checkpoint_sha256']==sha(row['checkpoint'])
            assert artifact['target_split']=='TRAIN' and not artifact['evaluation_data_used_for_fit']
            assert provenance['TRAIN_scenes']==8192 and provenance['fit_generation_seed']==6817
            assert not provenance['smoke_only'] and provenance['fit_device']=='cuda'
            assert sha(calibration.parent/'train_sizes.pt')==provenance['train_sizes_sha256']
            assert sha(calibration.parent/'train_files.pkl')==provenance['selected_TRAIN_manifest_sha256']
            assert report['num_samples']==report['num_gt_samples']==row['num_samples']
            assert report['reference_mode']=='matched' and report['map_source']=='reference'
            assert report['runtime']['trainer_precision']=='32-true' and report['runtime']['float32_matmul_precision']=='medium'
            assert report['decoder']['sampling_steps']==20 and report['decoder']['ema_num_updates']==row['budget']
            if row['num_samples']==50000:assert report['full_membership']
            for index,name in enumerate(METRICS):
                value=divergence(hist[name],10 if index<2 else 100)['value']
                assert math.isclose(value,row[name],rel_tol=1e-9,abs_tol=1e-10)
                assert math.isclose(value,report['agent_metrics'][name],rel_tol=1e-9,abs_tol=1e-10)
                assert hist[name]['matched_gt']==raw_hist[name]['matched_gt']
            value=100*report['num_colliding_vehicles']/report['num_generated_vehicles']
            assert math.isclose(value,row['collision_rate'],rel_tol=1e-9,abs_tol=1e-10)
            counts_equal=all(report[k]==raw_report[k] for k in ('num_generated_vehicles','num_gt_vehicles','num_generated_onroad_vehicles'))
            untouched_equal=all(hist[k]['generated']==raw_hist[k]['generated'] for k in UNAFFECTED)
            cells.append(dict(GEN=row['eval_seed'],scenes=row['num_samples'],budget=row['budget'],
                report_sha256=sha(row['metrics_path']),calibration_sha256=sha(calibration),
                counts_unchanged=counts_equal,non_size_histograms_exactly_unchanged=untouched_equal,
                non_size_metric_deltas={k:row[k]-baseline[k] for k in UNAFFECTED},
                interpretation='If non-size counts or histograms differ, investigate same-GPU reproducibility before claiming pure physical-size isolation.'))
    result=dict(status=('PASS' if all(c['counts_unchanged'] and c['non_size_histograms_exactly_unchanged'] for c in cells)
                        else 'NATIVE_METRICS_PASS_OUTPUT_ISOLATION_REQUIRES_REVIEW') if cells else 'WAITING_FOR_ACTUAL_EVALUATION',
        at_utc=now(),auditor_sha256=sha(__file__),protocol_sha256=sha(ART/'protocol.json'),comparison_sha256=comparisons,cells=cells)
    save(ART/'native_metric_audit.json',result);print(result['status'],len(cells))


if __name__=='__main__':main()
