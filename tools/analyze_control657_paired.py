"""Audit and re-score existing overnight observations against fresh paired657.

Exact exposure, scene membership, TRAIN/GEN seeds, EMA and semantic cohort must
match. Complete stages only. Native JSD values and raw values are both retained.
"""
from collections import defaultdict
from datetime import datetime, timezone
import argparse,csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import analyze_frozen_map_search as old
from tools import overnight_search_analysis as overnight
from tools.run_control657_paired import ART,RUNS,HISTORY
METRICS=old.METRICS;ATTR=old.ATTR
OUT=ROOT/'output/reports/control657_paired_20261012'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def build(partial=False):
    state=read(ART/'driver_status.json')
    if not partial and state['status']!='COMPLETE': raise RuntimeError('All paired stages must complete before final scoring')
    protocol=read(ART/'paired_protocol.json');prior=read(HISTORY)
    assert protocol['history_sha256']==sha(HISTORY)
    assert read(ART/'legacy_compatibility.json')['status']=='PASS'
    signature=lambda r:(r['eval_manifest_sha256'],r['seen_training_examples'],r['seed'],r['eval_seed'])
    old_controls={signature(r):r for r in prior if r['variant']=='control'}
    current=[];auxiliary=[];evidence=[];warnings=[]
    for spec in protocol['specs']:
        run=RUNS/spec['label'];complete=run/'verified_completion.json'
        if not complete.is_file():
            if partial: continue
            raise RuntimeError('Unverified stage '+spec['label'])
        completed=read(complete);assert completed['status']=='VERIFIED_COMPLETE' and completed['returncode']==0
        source=run/'summary.json';assert completed['summary_sha256']==sha(source)
        assert 'RUNTIME_LOADER_SUCCESS' in (run/'console.log').read_text()
        native=read(source);p=read(run/'protocol.json')
        method=old.digest(protocol['source_files'])
        execution={'source_hashes':protocol['source_files'],'command':spec['command']}
        for row in native:
            enriched=old.enrich(row,p,run,source,warnings)
            if enriched is None: raise RuntimeError('Invalid observed cell')
            converted=overnight.attach_cohorts(enriched,p,execution,method)
            original=old_controls.get(signature(converted))
            converted.update(stage=spec['label'],model_data_source_sha256=method,
                             model_data_source_files=protocol['source_files'],variant='control657',
                             planned_generation_seeds=[2817] if spec['scenes']==1024 else [2817,3817],
                             initial_generator_sha256=row['initial_generator_sha256'],initial_map_sha256=row['initial_map_sha256'],
                             checkpoint_sha256=sha(row['checkpoint']),metrics_sha256=sha(row['metrics_path']))
            if original:
                assert converted['cohort_sha256']==original['cohort_sha256'],[(k,converted['cohort'].get(k),original['cohort'].get(k)) for k in converted['cohort'] if converted['cohort'].get(k)!=original['cohort'].get(k)]
                for key in ('calibration_sha256','map_checkpoint_sha256','initial_generator_sha256','initial_map_sha256'):
                    assert converted[key]==original[key],(key,converted.get(key),original.get(key))
                assert row['train_order_sha256']==read(Path(original['run_path'])/'protocol.json')['train_order_sha256']
                converted['phase']=original['phase'];current.append(converted)
            else:
                converted['phase']='auxiliary';auxiliary.append(converted)
        evidence.append({'stage':spec['label'],'summary_sha256':sha(source),'protocol_sha256':sha(run/'protocol.json'),'completion_sha256':sha(complete)})
    keys=[signature(r) for r in current];assert len(set(keys))==len(keys),'Duplicate new control cells'
    base={signature(r):r for r in current}
    grouped=defaultdict(list)
    for row in prior:
        grouped[(row['phase'],row['cohort_sha256'],row['seen_training_examples'],row['variant'])].append(row)
    groups=defaultdict(list);missing=[]
    for (phase,cohort,exposure,name),rows in grouped.items():
        if any(signature(r) not in base for r in rows):
            missing.append({'phase':phase,'variant':name,'exposure':exposure,'status':'MATCHED_CONTROL_PENDING'});continue
        controls=[base[signature(r)] for r in rows]
        assert len({r['cohort_sha256'] for r in rows+controls})==1
        score=overnight.compare(rows,controls)
        score.update(variant=name,label='Old control' if name=='control' else name,
                     exposure=exposure,equivalent_epochs=exposure/493542,phase=phase,
                     train_seeds=sorted({r['seed'] for r in rows}),generation_seeds=sorted({r['eval_seed'] for r in rows}),
                     candidate_observations=len(rows),control_observations=len(controls),
                     candidate_values_raw={k:score['metrics'][k]/old.MULTIPLIERS[k] for k in METRICS},
                     control_values_native=overnight.controller.aggregate(controls),
                     control_values_raw={k:overnight.controller.aggregate(controls)[k]/old.MULTIPLIERS[k] for k in METRICS},
                     status='BEATS_657' if score['ranking_eligible'] and score['R']<1 else 'DOES_NOT_QUALIFY',
                     changed_percent={k:100*(score['ratios'][k]-1) for k in METRICS})
        for seed,s in score['per_training_seed'].items():
            s['status']='BEATS_657' if s['eligible'] and s['R']<1 else 'DOES_NOT_QUALIFY'
        groups[(phase,cohort,exposure)].append(score)
    result={'scope':'Fresh two-TRAIN paired657 under original frozen-map quick-search protocol; no128epoch conclusion',
            'generated_utc':datetime.now(timezone.utc).isoformat(),'complete':state['status']=='COMPLETE' and not missing,
            'control_commit':protocol['control_commit'],'original_observations':len(prior),
            'matched_new_control_observations':len(current),'auxiliary_observations':len(auxiliary),
            'new_training_seeds':[4817,5817],'final_exposure_per_TRAIN':5242620,
            'definition':'Mean of7 candidate/control arm-mean ratios; GEN averaged within TRAIN, then TRAIN means weighted equally',
            'attribute_definition':'Mean of length,width,speed candidate/control arm-mean ratios',
            'gates':'L/W/speed each<=1.15; onroad decline<=.02; generated vehicle / GT vehicle in [.9,1.1]; pooled and every TRAIN must pass',
            'compatibility':read(ART/'legacy_compatibility.json'),'protocol':protocol,'source_evidence':evidence,
            'groups':[{'phase':k[0],'cohort':k[1],'exposure':k[2],'equivalent_epochs':k[2]/493542,
                       'arms':sorted(v,key=lambda s:s['R'])} for k,v in sorted(groups.items())],
            'missing':missing,'warnings':warnings,'limitations':[
                '657 recipe transplanted into fixed-map, TRAIN-normalizer, medium-matmul protocol; native657128epoch run differs.',
                'All candidates reuse historical saved observations. Only the control is newly trained; selection is retrospective.',
                'Early baseline checkpoints reuse the two fresh long trajectories. They are not new independent TRAIN models.',
                'Screen has one TRAIN and one GEN; medium has one TRAIN and two GEN; final has two TRAIN and two GEN.',
                '5817 compares only the previously frozen separate_speed recipe and old control; no5817 evidence for other screen arms.',
                '4096 and full50k sets were already used in selection/audits; not pristine unseen test.',
                'Equal exposure is not equal optimizer updates or wall clock; B256 doubles updates relative to B512.',
                'Runtime GC/load policies are disclosed separately; no bitwise GPU trajectory claim.',
                'Model/loss changes are combined; comparisons do not isolate a single causal hyperparameter.',
                '10.6224epoch scores do not establish128epoch convergence ranking.']}
    if not partial: assert result['complete'],missing
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'comparison.json',result);save(OUT/'new_control_observations.json',current)
    save(OUT/'auxiliary_observations.json',auxiliary)
    columns=['phase','exposure','equivalent_epochs','variant','train_seeds','generation_seeds','R','R_attr','status','pooled_gates_pass','each_training_seed_gates_pass',*METRICS]
    with (OUT/'rankings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for g in result['groups']:
            for a in g['arms']:writer.writerow({**{k:a[k] for k in columns if k not in METRICS},**{k:a['metrics'][k] for k in METRICS}})
    print(json.dumps({'complete':result['complete'],'new_control_cells':len(current),'missing':len(missing),'groups':[(g['phase'],g['exposure'],[(a['variant'],round(a['R'],4),round(a['R_attr'],4),a['status']) for a in g['arms']]) for g in result['groups']]},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');build(p.parse_args().partial)
