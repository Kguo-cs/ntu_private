"""Recover an externally interrupted first-round stage without repeating completed arms."""
import argparse
import csv
import math
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_init_diffusion_657_iteration as iteration
from tools.run_init_diffusion_657_iteration import ART,RUNS,read,save,sha,stage,now


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    folder=RUNS/'medium';archived=ART/'interrupted_medium_observations.json'
    if not archived.exists():
        rows=read(folder/'summary.json')
        save(archived,rows)
        save(ART/'first_round_interruption.json',{'cause':'external turn interruption','final_logged_combination_step':3312,
            'original_stage_exit_success':False,'preserved_observations_sha256':sha(archived),
            'original_console_sha256':sha(folder/'console.log'),'interrupted_utc':now()})
    rows=read(archived)
    expected={(n,b,g) for n in ('train_features','lane_centers','depth5') for b in (2048,4096) for g in (2817,3817)}
    assert len(rows)==len(expected) and {(r['variant'],r['budget'],r['eval_seed']) for r in rows}==expected
    evidence=[]
    for name in ('train_features','lane_centers','depth5'):
        trained=read(folder/name/'train_summary.json');assert trained['steps']==4096
        assert all(math.isfinite(float(v)) for k,v in trained['last_window'].items())
        assert trained['last_window']['map_grad_norm_before_clip']==0
        evidence.append({'arm':name,'train_summary_sha256':sha(folder/name/'train_summary.json'),
            'last_checkpoint_sha256':sha(folder/name/'last.ckpt'),'config_sha256':sha(folder/name/'config.yaml')})
    for r in rows:
        report=read(r['metrics_path']);assert report['num_samples']==report['num_gt_samples']==4096
        assert r['evaluation_weights']=='ema' and r['seed']==4817 and r['map_frozen']
        for k,v in report['agent_metrics'].items():assert float(v)==r[k]
        evidence.append({'metrics_path':r['metrics_path'],'sha256':sha(r['metrics_path'])})
    save(ART/'first_round_recovery_protocol.json',{'created_utc':now(),'fresh_combination_restart':True,
        'resume_interrupted_generator':False,'reuse_completed_observations':True,'source_sha256':sha(__file__),
        'evidence':evidence,'original_interruption_record_sha256':sha(ART/'first_round_interruption.json')})
    if not args.execute:print('Validated preserved observations:',len(rows));return
    combo=stage('medium_combo_recovery',['adaptive_combo'],4096,4096,checkpoints=(2048,),gens=(2817,3817))
    previous=read(folder/'adaptive_combo/initialization.json')
    assert combo[0]['initial_generator_sha256']==previous['initial_generator_sha256']
    combined=rows+combo
    save(folder/'summary.json',combined)
    fields=list(dict.fromkeys(k for r in combined for k in r))
    with (folder/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    save(folder/'verified_completion.json',{'status':'VERIFIED_RECOVERED_COMPLETE','original_stage_exit_success':False,
        'recovery_stage_returncode':0,'finished_utc':now(),'summary_sha256':sha(folder/'summary.json'),
        'observations':len(combined),'recovery_protocol_sha256':sha(ART/'first_round_recovery_protocol.json'),
        'recovery_stage_completion_sha256':sha(RUNS/'medium_combo_recovery/verified_completion.json')})
    print('RECOVERED medium:',len(combined),flush=True)
    # Reuse the original, preregistered first-round selection and confirmation.
    iteration.main()


if __name__=='__main__':main()
