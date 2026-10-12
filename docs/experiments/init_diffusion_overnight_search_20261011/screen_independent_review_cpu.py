"""Independent stdlib-only actual screen review. Never chooses a new arm."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path('/home/ke/code/sim')
ART=ROOT/'docs/experiments/init_diffusion_overnight_search_20261011'
RUNS=ROOT/'logs/init_diffusion_overnight_20261011'
METRICS=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
ATTR=('speed_jsd','length_jsd','width_jsd')
TRAIN=493542

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def exposure(step,batch):
    epochs,remainder=divmod(step,(TRAIN+batch-1)//batch)
    return epochs*TRAIN+remainder*batch

def physical_cohort(row,report,protocol,execution):
    return {'source_hashes':execution['source_hashes'],
            'train_order_sha256':row['train_order_sha256'],
            'calibration_sha256':row['calibration_sha256'],
            'initial_map_sha256':row['initial_map_sha256'],
            'map_checkpoint_sha256':row['map_checkpoint_sha256'],
            'eval_manifest_sha256':row['eval_manifest_sha256'],
            'reference_mode':report['reference_mode'],
            'reference_source_manifest_sha256':report['reference_source_manifest_sha256'],
            'official_source_sha256':report['official_source_sha256'],
            'map_source':report['map_source'],
            'conditioning':report['decoder']['conditioning'],
            'evaluation_weights':row['evaluation_weights'],
            'TRAIN_seed':row['seed'],'GEN_seed':row['eval_seed'],
            'scenes':row['num_samples'],
            'runtime':{k:protocol.get(k) for k in ('defer_cyclic_gc','young_gc_interval','cyclic_gc_policy','workers','precision','float32_matmul_precision','torch_version','gpu')}}

def completed_stages():
    stages,observations,warnings=[],{},[]
    for p in sorted(RUNS.glob('screen_*/execution.json')):
        execution=read(p); stage=p.parent
        status={'stage':stage.name,'status':execution.get('status'),'included':False}
        if status['status']!='success' or execution.get('returncode')!=0:
            stages.append(status);continue
        try:
            spec=execution['completion_spec'];protocol=read(stage/'protocol.json')
            frozen=read(ART/'source_freeze.json')['source_hashes']
            assert execution['source_hashes']==frozen,'Stage source SHA differs from final source freeze'
            for relative,recorded in frozen.items():
                assert sha(ROOT/relative)==recorded,'Live source SHA changed after freeze: '+relative
            assert spec['seed']==4817 and spec['scenes']==1024 and spec['generations']==[2817], 'Unplanned screen TRAIN/GEN/cohort'
            assert protocol['reference_mode']=='matched' and protocol['require_full_set'] is False, 'Wrong metric scope'
            assert protocol['workers']==0 and protocol['precision']=='32-true', 'Wrong loader/precision'
            assert protocol['defer_cyclic_gc'] is True and protocol['young_gc_interval']==128, 'Wrong runtime policy'
            assert set(spec['budgets'])=={spec['steps']//2,spec['steps']}, 'Missing actual midpoint/endpoint plan'
            assert (spec['steps'],spec['batch']) in ((768,512),(1536,256)), 'Wrong screening budget'
            for entry in execution['artifacts'].values():assert sha(Path(entry['path']))==entry['sha256'],'Artifact hash changed'
            log=RUNS/(stage.name+'.log')
            assert log.read_text().rstrip().splitlines()[-1]=='DONE '+str(stage), 'No terminal DONE'
            rows=read(stage/'summary.json');assert len(rows)==2,'Incomplete screen rows'
            assert {row['budget'] for row in rows}==set(spec['budgets']),'Missing/duplicate checkpoint cell'
            converted=[]
            for row in rows:
                assert row['variant']==spec['name'] and row['seed']==4817 and row['eval_seed']==2817,'Unexpected arm/TRAIN/GEN'
                assert row['seen_training_examples']==exposure(row['budget'],spec['batch']),'Exposure incorrect'
                assert row['evaluation_weights']=='ema' and row['num_samples']==1024,'Wrong weights/scope'
                assert row['map_frozen'] is True and row['initialization_scope']=='frozen_pretrained_map_fresh_G1','Wrong map/training origin'
                assert row['actual_train_offset']==0 and row['resumed_from'] is None and not row['optimizer_continued'],'Not fresh training'
                assert row['resume_global_step']==0,'Unexpected resumed steps'
                report=read(Path(row['metrics_path']))
                assert report['num_samples']==1024 and report['num_gt_samples']==1024,'Wrong report scene count'
                assert report['reference_mode']=='matched' and report['runtime']['torch_initial_seed']==2817,'Wrong matched GEN report'
                assert report['decoder']['use_ema'] is True,'Not EMA evaluation'
                assert report['decoder']['ema_num_updates']==row['budget'],'Wrong EMA exposure'
                assert report['map_source']=='reference','Not map-conditioned reference map evaluation'
                for metric in METRICS:
                    assert math.isfinite(row[metric]) and row[metric]>=0,'Invalid metric'
                    assert math.isclose(row[metric],report['agent_metrics'][metric],rel_tol=1e-10,abs_tol=1e-12),'Summary/report mismatch'
                converted.append({'row':row,'cohort':physical_cohort(row,report,protocol,execution),'report':report})
            observations[spec['name']]=converted
            status.update(included=True,midpoint_actual_exposure=196608,endpoint_actual_exposure=393216,
                          TRAIN=4817,GEN=2817,independent_training_N=1,
                          peak_rss_kib=execution.get('peak_observed_rss_kib'),
                          source_freeze_sha256=sha(ART/'source_freeze.json'),
                          source_hashes_match_final_freeze=True)
        except (OSError,KeyError,TypeError,ValueError,AssertionError) as err:
            status['exclusion_reason']=str(err);warnings.append(stage.name+': '+str(err))
        stages.append(status)
    return stages,observations,warnings

def comparison(candidate,baseline):
    ratios={metric:candidate[metric]/baseline[metric] for metric in METRICS}
    road_delta=candidate['generated_onroad_fraction']-baseline['generated_onroad_fraction']
    count_ratio=candidate['num_generated_vehicles']/candidate['num_gt_vehicles']
    gate={'onroad':road_delta>=-.02,'vehicle_count':.9<=count_ratio<=1.1,
          **{m:ratios[m]<=1.15 for m in ATTR}}
    return {'R_seven':statistics.fmean(ratios.values()),'R_speed_shape':statistics.fmean(ratios[m] for m in ATTR),
            'metric_ratios':ratios,'onroad_delta_percentage_points':100*road_delta,
            'generated_vehicle_to_matchedGT_ratio':count_ratio,'gate_checks':gate,'gate_pass':all(gate.values()),
            'attributes_vs_geometry':{'speed_shape_mean_ratio':statistics.fmean(ratios[m] for m in ATTR),
                                     'lat_ang_mean_ratio':statistics.fmean(ratios[m] for m in ('lat_dev_jsd','ang_dev_jsd')),
                                     'collision_ratio':ratios['collision_rate'],
                                     'nearest_distance_ratio':ratios['nearest_dist_jsd']}}

def rank_values(values):
    ranked=sorted((value,name) for name,value in values.items());ranks={}
    for val,group in itertools.groupby(enumerate(ranked,1),lambda pair:pair[1][0]):
        cells=list(group);r=statistics.fmean(index for index,_ in cells)
        ranks.update({name:r for _,(_,name) in cells})
    return ranks

def stability(cells):
    novel={name:values for name,values in cells.items() if name!='control'}
    if len(novel)<2:return {'status':'insufficient_common_completed_arms','common_arm_count':len(novel)}
    mid={name:values['midpoint']['R_seven'] for name,values in novel.items()}
    end={name:values['endpoint']['R_seven'] for name,values in novel.items()}
    rm,re=rank_values(mid),rank_values(end)
    names=sorted(novel);xm,xe=statistics.fmean(rm.values()),statistics.fmean(re.values())
    cov=sum((rm[n]-xm)*(re[n]-xe) for n in names)
    den=math.sqrt(sum((rm[n]-xm)**2 for n in names)*sum((re[n]-xe)**2 for n in names))
    inversions=[];ties=0
    for a,b in itertools.combinations(names,2):
        d0,d1=mid[a]-mid[b],end[a]-end[b]
        if d0*d1<0:inversions.append([a,b])
        elif d0*d1==0:ties+=1
    return {'status':'actual_diagnostic_rank_comparison','common_arm_count':len(names),
            'R_spearman':cov/den if den else None,'pair_order_inversions':len(inversions),'inverted_pairs':inversions,
            'pair_count':len(names)*(len(names)-1)//2,'tied_pairs':ties,
            'gate_flips':[{'arm':name,'midpoint_gate':novel[name]['midpoint']['gate_pass'],'endpoint_gate':novel[name]['endpoint']['gate_pass']}
                          for name in names if novel[name]['midpoint']['gate_pass']!=novel[name]['endpoint']['gate_pass']],
            'midpoint_R_order':sorted(names,key=mid.get),'endpoint_R_order':sorted(names,key=end.get),
            'interpretation':'Descriptive stability between 0.3984 and 0.7967 epochs on the same adaptive TRAIN4817/GEN2817/1024 scenes. It does not forecast final training rank or establish causality.'}

def main():
    stages,obs,warnings=completed_stages();cells={};cohorts={};mismatch=[]
    control=obs.get('control')
    if control:
        for name,records in obs.items():
            arm={}
            for period,exp in (('midpoint',196608),('endpoint',393216)):
                candidate=next(r for r in records if r['row']['seen_training_examples']==exp)
                base=next(r for r in control if r['row']['seen_training_examples']==exp)
                if candidate['cohort']!=base['cohort']:
                    mismatch.append({'arm':name,'period':period,'differing_keys':[k for k in base['cohort'] if candidate['cohort'].get(k)!=base['cohort'][k]]});continue
                cohorts[name]=key(candidate['cohort'])
                arm[period]=comparison(candidate['row'],base['row'])
                arm[period]['native_metrics']={m:candidate['row'][m] for m in METRICS}
            if set(arm)=={'midpoint','endpoint'}:
                arm['ratio_change_end_minus_mid']={m:arm['endpoint']['metric_ratios'][m]-arm['midpoint']['metric_ratios'][m] for m in METRICS}
                arm['initial_generator_sha256']=records[0]['row']['initial_generator_sha256']
                cells[name]=arm
    anchor=obs.get('combo_reference')
    if anchor:
        for name,records in obs.items():
            if name not in cells:continue
            r=next(r['row'] for r in records if r['row']['seen_training_examples']==393216)
            b=next(r['row'] for r in anchor if r['row']['seen_training_examples']==393216)
            cells[name]['endpoint_relative_combo_reference']=comparison(r,b)
    review={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'status':'ACTUAL_PARTIAL_SCREEN_DIAGNOSTIC' if cells else 'WAITING_FOR_VERIFIED_SCREEN_ENDPOINTS',
            'cpu_only':True,'GPU_used':False,'training_sources_modified':False,'new_selection_made':False,
            'stages':stages,'actual_complete_arm_count':len(cells),'planned_arm_count':18,
            'verified_conditions':{'same_sources_runtime_cohort_gen':bool(cells) and not mismatch,
                                   'midpoint_exposure':196608,'endpoint_exposure':393216,
                                   'midpoint_full_epochs':196608/TRAIN,'endpoint_full_epochs':393216/TRAIN,
                                   'TRAIN_seed':4817,'GEN_seed':2817,'independent_training_N':1,'evaluation_scene_count':1024},
            'cohort_hashes':cohorts,'mismatches':mismatch,'observed_comparisons':cells,
            'rank_stability':stability(cells),'warnings':warnings,
            'confidence_limitations':['Only successful stages with verified exact artifact hashes and terminal DONE count.',
                                     'Single reused TRAIN and single GEN do not provide training-seed uncertainty or significance.',
                                     'All screening evaluations reuse the same 1024 subset; selection optimism is expected.',
                                     'New novel arms inherit combo_reference; differences versus control include batch and multiple existing settings.',
                                     'Architecture/new modules can consume constructor RNG differently; equal seed is not identical parameter initialization.',
                                     'Comparisons versus combo_reference are descriptive and cannot alone prove a causal mechanism.',
                                     'No incomplete endpoint is padded or used to choose a candidate. Frozen controller owns all selection.'],
            'method_script':str(Path(__file__)), 'method_script_sha256':sha(Path(__file__))}
    target=ART/'screen_independent_review.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');temporary.replace(target)
    print(json.dumps({'completed':len(cells),'stages':len(stages),'mismatches':mismatch,'warnings':warnings}))

if __name__=='__main__':main()
