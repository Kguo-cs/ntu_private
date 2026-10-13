"""Independent histogram/count/NFE/file QA of completed native TEST cells."""
import hashlib
import json
from pathlib import Path
import math
from scipy.spatial.distance import jensenshannon
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_evening_20261013'
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    checks=0;allrows={};sources={}
    for phase in ('n20_sampler','guidance_full50000','guidance_final_only50000'):
        protocol=BASE/phase/'protocol.json';sources[str(protocol.relative_to(ROOT))]=sha(protocol)
        for row in read(BASE/phase/'observations.json'):
            allrows[row['folder']]=row
    cohorts={}
    for folder,row in allrows.items():
        folder=Path(folder);proof=read(folder/'verified_completion.json')
        assert proof['status']=='VERIFIED_COMPLETE';checks+=1
        for name,value in proof['artifact_sha256'].items():assert sha(folder/name)==value;checks+=1
        assert sha(folder/'cell.json')==proof['source_cell_sha256'];checks+=1
        report=read(row['native_report']);audit=read(row['sampler_audit']);hist=read(folder/'feature_histograms.json')
        assert report['num_samples']==report['num_gt_samples']==row['scenes'];checks+=1
        assert report['map_source']=='reference' and report['reference_mode']=='matched';checks+=1
        assert report['unverified_generation_timestep_count']==0;checks+=1
        assert audit['denoiser_parameters']==8083978 and audit['EMA_num_updates']==16384;checks+=1
        assert audit['training_loss_config_unchanged'] and set(audit['observed_NFE_per_sample'])=={20};checks+=1
        assert audit['total_denoiser_calls']==20*audit['sample_calls'];checks+=1
        assert report['runtime']['trainer_precision']=='32-true' and report['runtime']['float32_matmul_precision']=='medium';checks+=1
        for i,key in enumerate(('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd')):
            expected=float(jensenshannon(hist[key]['generated'],hist[key]['matched_gt'])**2*(10 if i<2 else 100))
            assert math.isclose(expected,row['metrics'][key],abs_tol=1.e-10), (folder,key,expected,row['metrics'][key]);checks+=1
            cohort=(row['scenes'],row['eval_files_sha256'],key)
            if cohort in cohorts:assert cohorts[cohort]==hist[key]['matched_gt'];checks+=1
            else:cohorts[cohort]=hist[key]['matched_gt']
        collision=100*report['num_colliding_vehicles']/report['num_generated_vehicles']
        assert math.isclose(collision,row['metrics']['collision_rate'],abs_tol=1.e-10);checks+=1
        if row['scenes']==50000:
            assert report['full_membership'];checks+=1
        if row['option']=='guide_final_only':
            assert audit['total_guidance_calls']==audit['sample_calls']==391;checks+=1
            assert audit['observed_guidance_calls_per_sample']==1;checks+=1
    result=dict(status='PASS',unique_actual_cells=len(allrows),checks=checks,source_protocol_sha256=sources,
                note='Historical source hashes retained in protocols; code later changed only in a separately registered shape phase. No claim that current files still match old source hashes.')
    (BASE/'native_test_QA.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
