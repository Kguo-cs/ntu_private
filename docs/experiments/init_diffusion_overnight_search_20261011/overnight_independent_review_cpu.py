"""Stdlib-only independent execution/scope/protocol audit; never trains/selects."""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path('/home/ke/code/sim')
ART=ROOT/'docs/experiments/init_diffusion_overnight_search_20261011'
RUNS=ROOT/'logs/init_diffusion_overnight_20261011'
METRICS=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
ATTR=('speed_jsd','length_jsd','width_jsd')
VALUE_KEYS=METRICS+('generated_onroad_fraction','num_generated_vehicles','num_gt_vehicles')
TRAIN=493542
CUTOFF=datetime(2026,10,12,1,35,tzinfo=timezone.utc).timestamp()


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check(condition,message):
    if not condition:raise ValueError(message)
def save(path,value):
    temporary=Path(path).with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def exposure(step,batch):
    epochs,remainder=divmod(step,(TRAIN+batch-1)//batch)
    return epochs*TRAIN+remainder*batch

def physical_cohort(row,report,protocol,execution):
    return {'source_hashes':execution['source_hashes'],
            **{k:row[k] for k in ('calibration_sha256','initial_map_sha256','map_checkpoint_sha256','eval_manifest_sha256','num_samples','evaluation_weights')},
            **{k:report[k] for k in ('reference_mode','reference_source_manifest_sha256','official_source_sha256','map_source','generation_task')},
            'conditioning':report['decoder']['conditioning'],
            'sampling_steps':report['decoder']['sampling_steps'],
            'runtime':{k:protocol.get(k) for k in ('defer_cyclic_gc','young_gc_interval','cyclic_gc_policy','workers','precision','float32_matmul_precision','torch_version','gpu')}}

def recovery_contract():
    path=ART/'runtime_recovery/protocol.json'
    if not path.exists():return None
    value=read(path)
    check(value['source_freeze_sha256']==sha(ART/'source_freeze.json'),'Recovery changed source freeze')
    check(value['frozen_long_protocol_sha256']==sha(ART/'frozen_long_protocol.json'),'Recovery changed frozen long budget')
    check(value['wrapper_sha256']==sha(value['wrapper']),'Recovery controller source changed')
    for source,digest in value.get('runtime_loader_source_hashes',{}).items():check(sha(source)==digest,'Recovery loader source changed')
    bridge=ART/'runtime_recovery/verification_bridge.json'
    if bridge.exists():
        receipt=read(bridge)
        check(receipt['original_recovery_protocol_sha256']==sha(path),'Bridge changed original recovery protocol')
        for source,digest in receipt['source_hashes'].items():check(sha(source)==digest,'Completion bridge source changed')
    check(value['runtime_change']['remaining_young_gc_interval']==0 and value['no_budget_recalculation'],'Recovery runtime/budget contract differs')
    return value

def comparable_cohort(cohort):
    # GC is a disclosed process/runtime intervention, never folded into model
    # equivalence. Only the explicit verified recovery contract allows this.
    if recovery_contract() is None:return cohort
    return {k:v for k,v in cohort.items() if k!='runtime'}

def seed_means(rows,required=(4817,5817),generations=(2817,3817)):
    check(rows,'No actual rows to aggregate')
    out={}
    check({r['seed'] for r in rows}==set(required),'Incomplete or unexpected TRAIN set')
    for seed in required:
        cells=[r for r in rows if r['seed']==seed]
        check(len(cells)==len(generations) and {r['eval_seed'] for r in cells}==set(generations),'Incomplete/duplicate GEN cells within TRAIN')
        out[seed]={k:statistics.fmean(r[k] for r in cells) for k in VALUE_KEYS}
    return out

def mean_training(seeds):return {k:statistics.fmean(v[k] for v in seeds.values()) for k in VALUE_KEYS}
def comparison(values,baseline):
    check(all(math.isfinite(values[k]) and values[k]>=0 and baseline[k]>0 for k in METRICS),'Invalid ratios')
    ratios={k:values[k]/baseline[k] for k in METRICS}
    gate={k:ratios[k]<=1.15 for k in ATTR}
    gate.update(onroad=values['generated_onroad_fraction']>=baseline['generated_onroad_fraction']-.02,
                vehicle_ratio=values['num_gt_vehicles']>0 and .9<=values['num_generated_vehicles']/values['num_gt_vehicles']<=1.1)
    return {'R':statistics.fmean(ratios.values()),'R_attr':statistics.fmean(ratios[k] for k in ATTR),
            'ratios':ratios,'gate_checks':gate,'eligible':all(gate.values()),
            'onroad_delta_percentage_points':100*(values['generated_onroad_fraction']-baseline['generated_onroad_fraction'])}

def paired_comparison(candidate,control,required=(4817,5817)):
    c,b=seed_means(candidate,required),seed_means(control,required)
    pooled=comparison(mean_training(c),mean_training(b))
    per={str(seed):comparison(c[seed],b[seed]) for seed in required}
    rvals=[r['R'] for r in per.values()]
    pooled.update(per_TRAIN=per,pooled_gates_pass=pooled['eligible'],each_TRAIN_gates_pass=all(v['eligible'] for v in per.values()),
                  eligible=pooled['eligible'] and all(v['eligible'] for v in per.values()),independent_training_N=len(required),
                  paired_R_mean=statistics.fmean(rvals),paired_R_sample_sd=statistics.stdev(rvals) if len(rvals)>1 else None,
                  averaging='GEN averaged inside TRAIN, then equal TRAIN weighting; pooled R is ratio of metric means')
    return pooled

def phase(label):return next((p for p in ('screen','medium','long','full50000') if label.startswith(p+'_')),None)

def pin_protocols():
    path=ART/'independent_protocol_pins.json';pins=read(path) if path.exists() else {}
    errors=[]
    for name in ('source_freeze','search_protocol','medium_protocol','frozen_long_protocol','final_selection'):
        p=ART/(name+'.json')
        if not p.exists():continue
        actual=sha(p)
        if name in pins and pins[name]['sha256']!=actual:
            allowed=False
            if name=='final_selection':
                archived=ART/'runtime_recovery/attempt1/final_selection.json'
                if archived.exists() and sha(archived)==pins[name]['sha256'] and read(archived).get('complete_matrix') is False and recovery_contract() is not None:
                    # The old incomplete terminal was a state snapshot. Its
                    # archived bytes remain pinned; only an actual complete
                    # recovered matrix becomes a new immutable decision pin.
                    allowed=True
                    if read(p).get('complete_matrix') is True:
                        pins[name]={'sha256':actual,'first_observed_utc':datetime.now(timezone.utc).isoformat(),
                                    'archived_incomplete_state_sha256':sha(archived),'recovery_ledger':str(ART/'runtime_recovery/protocol.json')}
            if not allowed:errors.append(name+' changed after independent first observation')
        if name not in pins:pins[name]={'sha256':actual,'first_observed_utc':datetime.now(timezone.utc).isoformat()}
    save(path,pins)
    return pins,errors

def stage_plan(execution,protocol,long_plan,medium_plan):
    label,name=execution['label'],execution['completion_spec']['name']
    spec=execution['completion_spec'];p=phase(label);batch=spec['batch'];steps=spec['steps']
    if p=='screen':
        check(label=='screen_'+name and spec['seed']==4817 and spec['scenes']==1024 and spec['generations']==[2817],'Unexpected screen plan')
        check((steps,batch) in ((768,512),(1536,256)),'Unequal screen training budget')
        check(set(spec['budgets'])=={steps//2,steps},'Screen planned budgets missing')
    elif p=='medium':
        check(medium_plan and name in medium_plan['promoted'],'Unpromoted medium arm')
        check(label=='medium_'+name and spec['seed']==4817 and spec['scenes']==4096 and spec['generations']==[2817,3817],'Unexpected medium plan')
        check((steps,batch) in ((4096,512),(8192,256)),'Unequal medium training budget')
        check(set(spec['budgets'])=={steps//2,steps} and exposure(steps,batch)==2097048,'Wrong actual medium exposure')
    elif p in ('long','full50000'):
        check(long_plan,'Long/full stage exists without frozen prior plan')
        check(name in ('control',long_plan['challenger']),'Arm not frozen before held-out TRAIN')
        check(batch==(512 if name=='control' else 256),'Unexpected frozen-arm batch')
        expected_steps=long_plan['final_budget_B512'] if batch==512 else long_plan['final_budget_B256']
        check(steps==expected_steps and exposure(steps,batch)==long_plan['actual_final_exposure_per_TRAIN'],'Wrong final actual exposure')
        check(spec['generations']==[2817,3817],'Missing planned final GENs')
        if p=='long':
            check(spec['seed'] in (4817,5817) and spec['scenes']==4096 and label==f'long_seed{spec["seed"]}_{name}','Unexpected long TRAIN/cohort')
            b512=long_plan['final_budget_B512'];expected={b512//3*512//batch,(b512*2//3)*512//batch,steps}
            check(set(spec['budgets'])==expected,'Final checkpoint exposures do not bridge B512/B256')
        else:
            check(spec['seed']==4817 and spec['scenes']==50000 and label=='full50000_seed4817_'+name,'Unexpected full50k plan')
            check(set(spec['budgets'])=={steps},'Full50k must audit one final checkpoint')
    else:raise ValueError('Unexpected canonical stage family')
    return p

def verified_stages():
    frozen=read(ART/'source_freeze.json')['source_hashes'];search=read(ART/'search_protocol.json')
    arms={a['name']:a for a in search['arms']}
    medium_plan=read(ART/'medium_protocol.json') if (ART/'medium_protocol.json').exists() else None
    long_plan=read(ART/'frozen_long_protocol.json') if (ART/'frozen_long_protocol.json').exists() else None
    stages,records,errors=[],[],[]
    for path in sorted(RUNS.glob('*/execution.json')):
        execution=read(path);run=path.parent;p=phase(run.name)
        if p is None:continue
        stage={'stage':run.name,'phase':p,'status':execution.get('status'),'included':False,
               'peak_rss_kib':execution.get('peak_observed_rss_kib')}
        if execution.get('status')!='success' or execution.get('returncode')!=0:
            stage['reason']='Only verified complete successful stages count; no partial checkpoint inference'
            stages.append(stage);continue
        try:
            spec=execution['completion_spec'];protocol=read(run/'protocol.json')
            check(execution['source_hashes']==frozen,'Source SHA differs from final freeze')
            for relative,recorded in frozen.items():check(sha(ROOT/relative)==recorded,'Live source SHA changed: '+relative)
            check(execution['arm']==arms[spec['name']],'Actual arm differs from frozen initial design')
            check(execution['label']==run.name,'Stage label mismatch')
            p=stage_plan(execution,protocol,long_plan,medium_plan)
            for item in execution['artifacts'].values():check(sha(item['path'])==item['sha256'],'Completion artifact hash changed')
            log_path=RUNS/(run.name+'.log')
            if execution.get('runtime_recovery',{}).get('runtime_loader_source_hashes'):
                if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
                from tools.runtime_completion import authenticated_log_view
                view,policy_path=authenticated_log_view(run,log_path)
                check(policy_path is not None and 'runtime_loader_policy' in execution['artifacts'],'Certified loader policy artifact missing')
                check(view.read_text().rstrip().splitlines()[-1]=='DONE '+str(run),'No correct authenticated terminal DONE')
            else:
                check(log_path.read_text().rstrip().splitlines()[-1]=='DONE '+str(run),'No correct terminal DONE')
            check(protocol['reference_mode']=='matched' and protocol['scenes']==spec['scenes'],'Wrong metric scope')
            check(protocol['seed']==spec['seed'] and protocol['batch_size']==spec['batch'],'Wrong TRAIN/batch')
            check([protocol['eval_seed']]+([protocol['extra_eval_seed']] if protocol['extra_eval_seed'] is not None else [])==spec['generations'],'Wrong GEN plan')
            recovery=recovery_contract()
            expected_gc=128
            if execution.get('runtime_recovery'):
                check(recovery is not None,'Recovery stage lacks explicit ledger')
                meta=execution['runtime_recovery']
                check(meta['wrapper_sha256']==recovery['wrapper_sha256'],'Recovery wrapper SHA mismatch')
                check(meta.get('runtime_loader_source_hashes',{})==recovery.get('runtime_loader_source_hashes',{}),'Recovery loader dependency SHA mismatch')
                expected_gc=0
            elif recovery is not None and p=='long':
                check(run.name=='long_seed4817_control','Only original completed control4817 can inherit manual GC')
            check(protocol['workers']==0 and protocol['precision']=='32-true' and protocol['defer_cyclic_gc'] and protocol['young_gc_interval']==expected_gc,'Runtime policy mismatch')
            check(bool(protocol['require_full_set'])==(p=='full50000'),'Full-set flag mismatch')
            rows=read(run/'summary.json');expected={(b,spec['seed'],g) for b in spec['budgets'] for g in spec['generations']}
            cells={(r['budget'],r['seed'],r['eval_seed']) for r in rows}
            check(len(rows)==len(expected) and cells==expected,'Missing/duplicate/unplanned budget/TRAIN/GEN cells')
            converted=[]
            for row in rows:
                check(row['variant']==spec['name'] and row['batch_size']==spec['batch'],'Wrong variant/batch')
                check(row['seen_training_examples']==exposure(row['budget'],spec['batch']),'Wrong actual exposure')
                check(row['evaluation_weights']=='ema' and row['num_samples']==spec['scenes'],'Wrong EMA/scope')
                check(row['map_frozen'] and row['initialization_scope']=='frozen_pretrained_map_fresh_G1','Frozen map origin missing')
                check(row['map_checkpoint_sha256']==protocol['map_checkpoint_sha256'],'Map source SHA mismatch')
                check(row['evaluation_only']==(p=='full50000'),'Unexpected training/evaluation-only status')
                if p!='full50000':
                    check(row['actual_train_offset']==0 and row['resumed_from'] is None and not row['optimizer_continued'] and row['resume_global_step']==0,'Promoted G1/optimizer/EMA was not fresh')
                else:
                    check(protocol['variants']=='' and protocol['evaluation_checkpoints'] and not protocol['resume_map'],'Full50k unexpectedly trains/resumes')
                    check(protocol['eval_manifest'] is None and protocol['eval_offset']==0,'Full50k silently selected a smaller/cohort slice')
                    mapping=read(protocol['evaluation_checkpoints']);check(mapping=={spec['name']:row['checkpoint']},'Full50k checkpoint mapping mismatch')
                report=read(row['metrics_path'])
                check(report['reference_mode']=='matched' and report['num_samples']==spec['scenes'] and report['num_gt_samples']==spec['scenes'],'Matched physical GT scope mismatch')
                check(report['require_full_set']==(p=='full50000') and report['full_membership']==(p=='full50000'),'Report full GT membership mismatch')
                check(report['map_source']=='reference' and report['decoder']['mode']=='lane_conditioned','Wrong map input/evaluation mode')
                check(report['runtime']['torch_initial_seed']==row['eval_seed'] and report['decoder']['use_ema'] and report['decoder']['ema_num_updates']==row['budget'],'Wrong GEN/EMA state')
                check(report['unverified_generation_timestep_count']==0,'Unverified generation timestep')
                for metric in METRICS:
                    check(math.isfinite(row[metric]) and row[metric]>=0,'Invalid metric')
                    check(math.isclose(row[metric],report['agent_metrics'][metric],rel_tol=1e-10,abs_tol=1e-12),'Summary metric/report mismatch')
                    if p=='full50000':
                        check(math.isclose(row[metric],report['full_reference_agent_metrics'][metric],rel_tol=1e-10,abs_tol=1e-12),
                              'Actual full membership has different matched/full reference metric')
                check(0<=row['generated_onroad_fraction']<=1 and row['num_gt_vehicles']>0,'Invalid physical counts')
                converted.append({'row':row,'phase':p,'stage':run.name,'cohort':physical_cohort(row,report,protocol,execution),
                                  'report':report,'spec':spec})
            records+=converted
            stage.update(included=True,actual_rows=len(rows),TRAIN=spec['seed'],GEN=spec['generations'],
                         endpoint_actual_exposure=exposure(spec['steps'],spec['batch']),budget_updates=spec['budgets'],
                         source_sha_verified=True,artifact_sha_DONE_verified=True,
                         actual_scene_scope=spec['scenes'],fresh_training=p!='full50000')
        except (OSError,KeyError,TypeError,ValueError,IndexError) as err:
            stage['reason']=str(err);errors.append(run.name+': '+str(err))
        stages.append(stage)
    # Paired TRAIN stages must repeat the exact same full seeded scene order.
    for seed in (4817,5817):
        hashes={r['row']['train_order_sha256'] for r in records if r['row']['seed']==seed}
        if len(hashes)>1:errors.append(f'TRAIN{seed} has different training-order fingerprints across arms/phases')
    # Fresh promotions of the same TRAIN/arm must initialize the same raw G1.
    for seed,name in itertools.product((4817,5817),arms):
        hashes={r['row']['initial_generator_sha256'] for r in records if r['row']['seed']==seed and r['row']['variant']==name and r['phase']!='full50000'}
        if len(hashes)>1:errors.append(f'TRAIN{seed}/{name} raw G1 initialization changed between fresh phases')
    return stages,records,errors

def endpoints(records,p):
    return [r for r in records if r['phase']==p and r['row']['budget']==r['spec']['steps']]

def medium_review(records):
    end=endpoints(records,'medium');arms=sorted({r['row']['variant'] for r in end});out={'actual_completed_arms':arms,'required_actual_exposure':2097048,'comparisons':{}}
    if 'control' not in arms:return out
    base=[r for r in end if r['row']['variant']=='control']
    for arm in arms:
        cand=[r for r in end if r['row']['variant']==arm]
        check({signature(r['cohort']) for r in cand+base}.__len__()==1,'Different medium physical cohorts')
        out['comparisons'][arm]=paired_comparison([r['row'] for r in cand],[r['row'] for r in base],required=(4817,))
    return out

def final_review(records,pins):
    path=ART/'frozen_long_protocol.json'
    if not path.exists():return {'status':'WAITING_FOR_FROZEN_LONG_PROTOCOL','matrix_complete':False,'materialization_ready':False}
    plan=read(path);arms=('control',plan['challenger']);end=endpoints(records,'long')
    expected={(arm,seed,gen) for arm in arms for seed in (4817,5817) for gen in (2817,3817)}
    cells={(r['row']['variant'],r['row']['seed'],r['row']['eval_seed']) for r in end}
    paired_seeds=[seed for seed in (4817,5817)
                  if {(arm,seed,gen) for arm in arms for gen in (2817,3817)}<=cells]
    out={'status':'ACTUAL_PARTIAL_FINAL_MATRIX','frozen_challenger':plan['challenger'],'matrix_complete':False,
         'actual_cells':sorted(cells),'required_cells':sorted(expected),'missing_cells':sorted(expected-cells),
         'actual_final_exposure_per_TRAIN':plan['actual_final_exposure_per_TRAIN'],
         'planned_TRAIN_seeds':[4817,5817],'actual_complete_paired_TRAIN_seeds':paired_seeds,
         'independent_training_N':len(paired_seeds),
         'GEN_per_TRAIN':2,'no_heldout_5817_reselection':True,'materialization_ready':False}
    check(cells<=expected and len(end)==len(cells),'Unexpected/duplicate final endpoint cells')
    if cells!=expected:return out
    check({r['row']['seen_training_examples'] for r in end}=={plan['actual_final_exposure_per_TRAIN']},'Unequal final exposure')
    check(len({signature(comparable_cohort(r['cohort'])) for r in end})==1,'Final physical cohort/source mismatch')
    if recovery_contract() is not None:
        out['runtime_comparability']='Inherited TRAIN4817 control GC128 versus recovered no manual GC; same frozen objective/model/data/exposure, runtime differs. TRAIN5817 both no manual GC.'
    rows={arm:[r['row'] for r in end if r['row']['variant']==arm] for arm in arms}
    cmp=paired_comparison(rows[plan['challenger']],rows['control'])
    out.update(status='VERIFIED_ACTUAL_COMPLETE_8_CELL_MATRIX',matrix_complete=True,comparison=cmp)
    selected_path=ART/'final_selection.json'
    if not selected_path.exists():return out
    declared=read(selected_path);selected=declared.get('selected')
    check(declared.get('complete_matrix') is True and declared['frozen_challenger']==plan['challenger'],'Controller final declaration incomplete/changed')
    check(selected in arms,'Declared native best was not in frozen pair')
    valid_improvement=cmp['eligible'] and cmp['R']<1
    check((selected==plan['challenger'])==valid_improvement,'Final selected arm disagrees with frozen paired gate/R rule')
    check({(r['variant'],r['seed'],r['eval_seed']) for r in declared['rows']}==expected and len(declared['rows'])==8,'Declared final rows missing/duplicate')
    for row in declared['rows']:
        actual=next(r['row'] for r in end if (r['row']['variant'],r['row']['seed'],r['row']['eval_seed'])==(row['variant'],row['seed'],row['eval_seed']))
        check(row==actual,'Final declaration rows differ from verified ledger')
    ranking=next(v for v in declared['ranking'] if v['name']==plan['challenger'])
    check(math.isclose(ranking['R'],cmp['R'],rel_tol=1e-12) and ranking['eligible']==cmp['eligible'],'Controller pooled gate/R disagree')
    for seed in ('4817','5817'):
        check(math.isclose(ranking['per_training_seed'][seed]['R'],cmp['per_TRAIN'][seed]['R'],rel_tol=1e-12)
              and ranking['per_training_seed'][seed]['eligible']==cmp['per_TRAIN'][seed]['eligible'],'Controller per-TRAIN gate/R disagree')
    out.update(declared_selected_arm=selected,selection_verified_without_new_reselection=True,
               materialization_ready=True,materialization_evidence='All8 endpoint cells + pooled/perTRAIN gates and frozen selection rule independently verified',
               selected_is_improvement=valid_improvement,final_selection_sha256=sha(selected_path))
    return out

def full_review(records,final):
    end=endpoints(records,'full50000');out={'status':'WAITING_OR_PARTIAL_FULL50K','actual_cells':len(end),'complete':False,'new_selection_made':False}
    if not end:return out
    check(final['matrix_complete'] and final.get('selection_verified_without_new_reselection'),'Full50k before verified/frozen final selection')
    arms=('control',final['frozen_challenger']);expected={(arm,4817,gen) for arm in arms for gen in (2817,3817)}
    cells={(r['row']['variant'],r['row']['seed'],r['row']['eval_seed']) for r in end}
    check(cells<=expected and len(end)==len(cells),'Unexpected/duplicate full50k endpoint')
    for item in end:
        row=item['row'];origin=next(r['row'] for r in endpoints(records,'long') if r['row']['variant']==row['variant'] and r['row']['seed']==4817 and r['row']['eval_seed']==2817)
        check(row['checkpoint']==origin['checkpoint'] and row['budget']==origin['budget'],'Full50k checkpoint lineage differs from frozen final TRAIN4817')
        check(row['seen_training_examples']==origin['seen_training_examples'],'Full50k changed training exposure')
        check(row['num_gt_vehicles']==362309 and row['num_gt_onroad_vehicles']==360099,
              'Full50k GT counts disagree with independently inventoried actual reference universe')
        check(item['report']['feature_counts_gt']==[358867,360099,360099,362309,362309,362309],
              'Full50k GT feature counts disagree with independently inventoried actual reference universe')
    if cells!=expected:return out
    check(len({signature(r['cohort']) for r in end})==1,'Full50k physical cohort/source/runtime differs')
    # Require actual 50000 unique filenames; glob reads names only, never scene payloads.
    expected_names=sorted(p.name for p in (ROOT/'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    check(len(expected_names)==50000,'Canonical full physical cache does not have50000 scenes')
    for stage in {r['stage'] for r in end}:
        names=(RUNS/stage/'eval_files.txt').read_text().splitlines()
        check(names==expected_names,'Full50k actual filename scope differs from canonical full sorted cache')
    hists=[read(Path(r['row']['metrics_path']).parent/'feature_histograms.json') for r in end]
    for feature in hists[0]:
        for hist in hists:
            check(hist[feature]['matched_gt']==hists[0][feature]['matched_gt'],'Different full matchedGT histogram across arms/GEN')
            check(hist[feature]['matched_gt']==hist[feature]['full_gt'],'Full membership has different matched/fullGT histogram')
            check(tuple(hist[feature][k] for k in ('minimum','maximum','bin_width'))==tuple(hists[0][feature][k] for k in ('minimum','maximum','bin_width')),'Histogram physical bins changed')
    base=[r['row'] for r in end if r['row']['variant']=='control'];cand=[r['row'] for r in end if r['row']['variant']==final['frozen_challenger']]
    out.update(status='VERIFIED_ACTUAL_FULL50K_4_CELL_SCOPE_AUDIT',complete=True,TRAIN_seed=4817,GEN_seeds=[2817,3817],
               independent_training_N=1,checkpoint_lineage_verified=True,full_scope_and_GT_histograms_verified=True,
               descriptive_comparison=paired_comparison(cand,base,required=(4817,)),
               frozen_selected_arm=final['declared_selected_arm'],no_reselection_verified=True,
               interpretation='Full50000 scope audit of TRAIN4817 checkpoints; not an independent TRAIN repetition or untouched test and does not reselect native best')
    return out

def forecast_review(records):
    path=ART/'frozen_long_protocol.json'
    if not path.exists():return {'status':'WAITING_FOR_FROZEN_BUDGET'}
    plan=read(path);rule=plan['final_budget_rule'];names=('control',plan['challenger']);rates={};eval_rates={}
    for name in names:
        cells=[r['row'] for r in records if r['phase']=='medium' and r['row']['variant']==name]
        endpoint_rows=[r for r in cells if r['budget']==max(v['budget'] for v in cells)]
        fits={r['train_seconds'] for r in endpoint_rows};check(len(fits)==1,'Fit timing repeated inconsistently')
        rates[name]=next(iter(fits))/endpoint_rows[0]['budget'];eval_rates[name]=max(r['eval_seconds']/r['num_samples'] for r in cells)
    check(rule['observed_medium_seconds_per_update']==rates and rule['observed_max_eval_seconds_per_scene']==eval_rates,'Final throughput projection not based on actual medium')
    check(rule['gpu_cutoff_unix']==CUTOFF,'GPU cutoff changed')
    for p in rule['projections']:
        fit=2*(p['control_B512_updates']*rates['control']+p['candidate_B256_updates']*rates[names[1]])
        evaluation=12*4096*sum(eval_rates.values());audit=max(1200.,2*50000*sum(eval_rates.values())*1.25)
        total=(fit+evaluation)*1.15+audit
        check(math.isclose(p['with_15percent_margin_and_measured_audit_seconds'],total,rel_tol=1e-12),'Forecast omits actual fit/24eval/full50k costs')
        check(p['actual_exposure_per_TRAIN']==exposure(p['control_B512_updates'],512)==exposure(p['candidate_B256_updates'],256),'Forecast final budgets not equal exposure')
        check(p['fits_gpu_cutoff']==(rule['decision_unix']+total<=CUTOFF),'Forecast deadline fit flag wrong')
    expected=next((p for p in rule['projections'] if p['fits_gpu_cutoff']),rule['projections'][-1])
    check(rule['chosen']==expected,'Not highest fitting shared budget')
    check(plan['final_budget_B512']==expected['control_B512_updates'] and plan['final_budget_B256']==expected['candidate_B256_updates'],'Frozen final steps differ from measured forecast')
    return {'status':'VERIFIED_ACTUAL_MEDIUM_THROUGHPUT_BUDGET_MATH','chosen_budget':expected,
            'deadline_risk':rule['deadline_risk'],'caveat':'Forecast arithmetic verified; remaining jobs can run slower than observed rates.'}

def main():
    with (ART/'independent_review.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pins,pin_errors=pin_protocols();stages,records,errors=verified_stages();errors+=pin_errors
        checks={}
        for name,func in (('medium',lambda:medium_review(records)),('forecast',lambda:forecast_review(records)),('final',lambda:final_review(records,pins))):
            try:checks[name]=func()
            except (OSError,KeyError,TypeError,ValueError,StopIteration) as err:checks[name]={'status':'AUDIT_ERROR','reason':str(err)};errors.append(name+': '+str(err))
        try:checks['full50000']=full_review(records,checks['final'])
        except (OSError,KeyError,TypeError,ValueError,StopIteration) as err:checks['full50000']={'status':'AUDIT_ERROR','reason':str(err)};errors.append('full50000: '+str(err))
        queue=read(ART/'queue_complete.json') if (ART/'queue_complete.json').exists() else None
        failures=[{'stage':s['stage'],'status':s['status']} for s in stages
                  if s['status'] in ('failed','deadline_interrupted')]
        if queue and queue.get('status')=='COMPLETE':
            if not (checks['final'].get('matrix_complete') and checks['full50000'].get('complete')):errors.append('Controller claims COMPLETE without independently complete8+4cells')
        fully_complete=(checks['final'].get('matrix_complete') and checks['full50000'].get('complete')
                        and queue and queue.get('status')=='COMPLETE')
        output={'updated_utc':datetime.now(timezone.utc).isoformat(),
                'status':'INDEPENDENT_AUDIT_ERROR' if errors else 'VERIFIED_ACTUAL_COMPLETE_STAGE_AUDIT' if fully_complete else 'VERIFIED_ACTUAL_PARTIAL_STAGE_SNAPSHOT',
                'CPU_only':True,'GPU_used':False,'sources_modified':False,'candidate_selection_made':False,
                'stages':stages,'actual_verified_observations':len(records),'actual_completed_stage_count':sum(s['included'] for s in stages),
                'protocol_pins':pins,'checks':checks,'errors':errors,'controller_queue_complete':queue,
                'runtime_failed_or_interrupted_stages':failures,
                'confidence':'Only actual verified completed stages; GEN mean inside TRAIN; no missing cells padded; final held5817 recipe frozen before training; full50k no reselection.',
                'method_script_sha256':sha(__file__)}
        save(ART/'overnight_independent_review.json',output)
        print(json.dumps({'actual_stages':output['actual_completed_stage_count'],'actual_rows':len(records),
                          'final_complete':checks['final'].get('matrix_complete'),'full50k_complete':checks['full50000'].get('complete'),'errors':errors}))

if __name__=='__main__':main()
