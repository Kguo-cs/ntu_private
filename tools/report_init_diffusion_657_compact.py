"""CPU-only combined report of verified main/compact N20 research evidence.

Never loads historical657 as a primary denominator or reads partial metrics.
Writing reports does not change frozen drivers, models or experiment artifacts.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import html,json,math,os,statistics,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_init_diffusion_657_iteration import METRICS,read,save,sha,score
from tools import report_init_diffusion_657_edges as main_report
from tools import run_init_diffusion_657_compact_overnight as compact_driver
MAIN_ART=ROOT/'output/reports/init_diffusion_657_edges_20261013'
MAIN_RUNS=ROOT/'logs/init_diffusion_657_edges_20261013'
ART=ROOT/'output/reports/init_diffusion_657_compact_20261013'
RUNS=ROOT/'logs/init_diffusion_657_compact_20261013'
TIERS=('screen','medium','long4096','full50k')
LABELS=('Near','Lat','Ang','Length','Width','Speed','Collision')
NATIVE_IDENTITY_FIELDS=('metric_scope','generation_task','map_source','reference_mode','reference_source_manifest_sha256','official_source_sha256','backend_scope')
PAIR_FIELDS=('batch_size','train_order_sha256','calibration_sha256','map_checkpoint_sha256','initial_map_sha256','evaluation_weights')


def maybe(path,default=None):return read(path) if Path(path).is_file() else default

def atomic_text(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
            prefix=path.name+'.',suffix='.tmp',delete=False) as handle:
        temporary=Path(handle.name);handle.write(value)
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def write_report_json(path,value):
    atomic_text(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def scope_key(row):
    return (row['seed'],row['eval_seed'],row['seen_training_examples'],
            row['eval_manifest_sha256'],row['num_samples'])

def tier(spec):
    if spec['scenes']==50000:return 'full50k'
    if spec['scenes']==4096:return 'long4096' if spec['steps']>=10240 else 'medium'
    if spec['scenes']==1024:return 'screen'
    raise ValueError('Unregistered evaluation scene scope')

def source_digest(sources):
    import hashlib
    return hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def collect(source,art,runs):
    protocol=maybe(art/'protocol.json')
    status=maybe(art/'rootdriver_status.json') or maybe(art/'status.json') or {'status':'NOT_STARTED'}
    result={'source':source,'artifact_dir':str(art),'run_dir':str(runs),'status':status,
        'protocol_sha256':sha(art/'protocol.json') if protocol else None,
        'protocol':protocol,'source_bundle_sha256':source_digest(protocol['sources']) if protocol else None,
        'verified_stages':[],'rows':[],'partials_not_evidence':[],'quarantined':[]}
    for folder in sorted(runs.glob('*')) if runs.is_dir() else []:
        if not folder.is_dir():continue
        if not (folder/'edge_verified_completion.json').is_file():
            result['partials_not_evidence'].append({'source':source,'stage':folder.name,
                'execution':maybe(folder/'edge_execution.json'),'metrics_usable':False})
            continue
        try:
            rows,marker,spec=main_report.validate_stage(folder)
            contract=read(folder/'edge_contract.json');legacy=read(folder/'verified_completion.json')
            if not protocol:raise ValueError('Verified stage has no registered protocol')
            if contract['sources']!=protocol['sources'] or contract['plan_sha256']!=protocol['plan_sha256']:
                raise ValueError('Stage source/plan differs from frozen protocol')
            if legacy['status']!='VERIFIED_COMPLETE' or legacy['returncode']!=0 or legacy['summary_sha256']!=sha(folder/'summary.json'):
                raise ValueError('Native child did not complete successfully')
            for path,digest in contract['sources'].items():
                if sha(ROOT/path)!=digest:raise ValueError('Frozen source SHA differs: '+path)
            if compact_driver.validate_observations(rows,spec)!=marker['metrics_reports']:
                raise ValueError('Native metadata or metrics SHA differs')
            budgets=[spec['steps']] if spec.get('evaluation') else spec['checkpoints']+[spec['steps']]
            expected={(spec['variant'],b,g) for b in budgets for g in spec['gens']}
            actual={(r['variant'],r['budget'],r['eval_seed']) for r in rows}
            if actual!=expected or len(rows)!=len(expected):raise ValueError('Registered TRAIN/GEN/budget membership differs')
            for row in rows:
                native=read(row['metrics_path'])
                if spec.get('evaluation') and row['checkpoint']!=spec['evaluation'].get(row['variant']):
                    raise ValueError('Evaluation checkpoint differs from registered contract')
                for key in ('generated_onroad_fraction','num_generated_vehicles','num_gt_vehicles',
                            'num_generated_onroad_vehicles','num_gt_onroad_vehicles','num_colliding_vehicles'):
                    if row.get(key)!=native.get(key):raise ValueError('Native gate/population field differs: '+key)
                if native['runtime']['torch_initial_seed']!=row['eval_seed']:raise ValueError('Native GEN seed differs')
                if native['runtime']['float32_matmul_precision']!='medium':raise ValueError('Native matmul precision differs')
                if native['decoder']['ema_num_updates']!=row['budget']:raise ValueError('Native EMA exposure differs')
                if native['unverified_generation_timestep_count']!=0:raise ValueError('Native generation membership unverified')
            stage={'source':source,'stage':folder.name,'tier':tier(spec),'spec':spec,
                'finished_utc':marker['finished_utc'],'summary_sha256':sha(folder/'summary.json'),
                'contract_sha256':sha(folder/'edge_contract.json'),'marker_sha256':sha(folder/'edge_verified_completion.json'),
                'source_bundle_sha256':source_digest(contract['sources']),
                'runtime':main_report.stage_runtime(folder,rows),'driver_elapsed_seconds':marker.get('elapsed_seconds'),
                'observations':len(rows)}
            stage['actual_generator_parameters']=sorted({r['generator_num_parameters'] for r in rows if r.get('generator_num_parameters') is not None})
            stage['actual_generator_trainable_parameters']=sorted({r['generator_trainable_parameters'] for r in rows if r.get('generator_trainable_parameters') is not None})
            stage['actual_frozen_map_parameters']=sorted({r['frozen_map_num_parameters'] for r in rows if r.get('frozen_map_num_parameters') is not None})
            stage['runtime_log_sha256']=sha(folder/'console.log') if (folder/'console.log').is_file() else None
            result['verified_stages'].append(stage)
            for row in rows:
                native=read(row['metrics_path'])
                result['rows'].append({**row,'source':source,'stage':folder.name,'tier':stage['tier'],
                    'finished_utc':stage['finished_utc'],'source_bundle_sha256':stage['source_bundle_sha256'],
                    'metrics_sha256':sha(row['metrics_path']),'summary_sha256':stage['summary_sha256'],
                    'native_metric_identity':{key:native.get(key) for key in NATIVE_IDENTITY_FIELDS},
                    'native_membership':{'full_membership':native['full_membership'],
                        'require_full_set':native['require_full_set'],'reference_mode':native['reference_mode'],
                        'reference_source_manifest_sha256':native['reference_source_manifest_sha256']}})
        except (ValueError,RuntimeError,KeyError,OSError,json.JSONDecodeError) as error:
            result['quarantined'].append({'source':source,'stage':folder.name,'reason':str(error),'metrics_usable':False})
    return result

def matching(values,candidates):
    candidates=sorted(candidates,key=lambda row:(row.get('finished_utc',''),row.get('source')=='compact'))
    available={scope_key(row):row for row in candidates}
    selected=[]
    for row in values:
        base=available.get(scope_key(row))
        if base is None:return None
        if any(row[key]!=base[key] for key in PAIR_FIELDS):return None
        if row.get('native_metric_identity')!=base.get('native_metric_identity'):return None
        if row['source']!=base['source'] and not row.get('cross_source_alignment',False):return None
        selected.append(base)
    return selected

def primary_controls(values,all_rows):
    """Main keeps its phase controls; compact prefers its registered controls."""
    source=values[0]['source']
    candidate_controls=[row for row in all_rows if row['variant']=='control657'
        and (source=='compact' or row['source']=='main')]
    selected=[]
    for row in values:
        same=[r for r in candidate_controls if scope_key(r)==scope_key(row)
              and all(r[key]==row[key] for key in PAIR_FIELDS)
              and r.get('native_metric_identity')==row.get('native_metric_identity')
              and (r['source']==row['source'] or row.get('cross_source_alignment',False))]
        if source=='main':same=[r for r in same if r['tier']==row['tier']]
        if row['tier']=='full50k':
            long_scope=[r for r in candidate_controls if r['seed']==row['seed'] and r['budget']==10240 and r['num_samples']==4096
                and r['eval_seed']==row['eval_seed'] and r['seen_training_examples']==row['seen_training_examples']
                and all(r[key]==row[key] for key in PAIR_FIELDS)
                and (r['source']==row['source'] or row.get('cross_source_alignment',False))]
            if source=='compact' and any(r['source']=='compact' for r in long_scope):
                long_scope=[r for r in long_scope if r['source']=='compact']
            long_scope=sorted(long_scope,key=lambda r:r.get('finished_utc',''))
            if not long_scope:return None
            checkpoint=long_scope[-1]['checkpoint'];same=[r for r in same if r['checkpoint']==checkpoint]
        if source=='compact' and any(r['source']=='compact' for r in same):
            same=[r for r in same if r['source']=='compact']
        if not same:return None
        selected.append(max(same,key=lambda r:r.get('finished_utc','')))
    return selected

def comparison(values,controls):
    if controls is None:return {'available':False,'reason':'No hash-verified fresh657/anchor with every exact TRAIN, GEN, exposure and scene scope'}
    try:
        result=score(values,controls)
        result.update(available=True,denominator_cells=[{'source':r['source'],'stage':r['stage'],
            'TRAIN':r['seed'],'GEN':r['eval_seed'],'checkpoint':r['checkpoint'],
            'metrics_path':r['metrics_path'],'metrics_sha256':r['metrics_sha256']} for r in controls])
        return result
    except (RuntimeError,ValueError,ZeroDivisionError) as error:return {'available':False,'reason':str(error)}

def same_fresh657_initializer(primary,direct_anchor):
    """Describe the common-denominator R condition, not an actual promotion.

    B is this candidate's exact fresh657 control; A is its matched initializer.
    Compact medium/long compares R(C/B) with min(1, R(A/B)). Direct R(C/A)
    and its gates are different diagnostics. Other requirements stay in the driver.
    """
    if not primary.get('available') or not direct_anchor.get('available'):
        return {'available':False,'reason':'Both exact fresh657 and direct initializer comparisons are required'}
    try:
        if any(set(primary[k])!=set(direct_anchor[k]) for k in ('TRAIN_seeds','GEN_seeds')):
            raise ValueError('Fresh657 and direct initializer TRAIN/GEN grids differ')
        if 'denominator_cells' in primary or 'denominator_cells' in direct_anchor:
            pairs=lambda result:{(r['TRAIN'],r['GEN']) for r in result['denominator_cells']}
            if pairs(primary)!=pairs(direct_anchor):
                raise ValueError('Fresh657 and direct initializer denominator cell grids differ')
        baseline=primary['control_metrics'];anchor=direct_anchor['control_metrics']
        if not all(math.isfinite(float(baseline[k])) and float(baseline[k])>0
                   and math.isfinite(float(anchor[k])) and float(anchor[k])>=0 for k in METRICS):
            raise ValueError('Nonfinite or invalid common-denominator metrics')
        ratios={k:anchor[k]/baseline[k] for k in METRICS}
        anchor_R=statistics.fmean(ratios.values());candidate_R=primary['R']
        if not math.isfinite(float(candidate_R)):raise ValueError('Nonfinite candidate R')
        threshold=min(1.,anchor_R)
        return {'available':True,'candidate_R':candidate_R,'initializer_R':anchor_R,
            'initializer_ratios':ratios,'R_threshold':threshold,
            'R_condition_pass':candidate_R<threshold,
            'fresh657_gates_pass':primary['all_TRAIN_gates_pass'],
            'denominator':'candidate exact fresh657 control',
            'fresh657_denominator_cells':primary.get('denominator_cells',[]),
            'initializer_denominator_cells':direct_anchor.get('denominator_cells',[]),
            'definition':'With complete matched A: R(C/B) < min(1, R(A/B)); gates relative to B. Direct R(C/A) and A gates are not the promotion rule.',
            'scope':'Numeric condition only; actual compact promotion also checks phase, slope, parameter cap and per-TRAIN/GEN requirements. When long has no complete matched A, the driver uses threshold 1 rather than a partial A. Full50k confirmation uses fresh657 without the initializer threshold.'}
    except (KeyError,TypeError,ValueError,ZeroDivisionError) as error:
        return {'available':False,'reason':str(error)}


def mean_metrics(values):
    seeds=sorted({r['seed'] for r in values})
    return {k:statistics.fmean(statistics.fmean(r[k] for r in values if r['seed']==seed) for seed in seeds) for k in METRICS}

def evidence_completeness(values):
    category=values[0]['tier'];required_train=[4817] if category in ('screen','medium') else [4817,5817]
    required_gen=[2817] if category=='screen' else [2817,3817]
    pairs={(r['seed'],r['eval_seed']) for r in values}
    required={(seed,gen) for seed in required_train for gen in required_gen}
    terminal=values[0]['budget']==({'screen':768,'medium':4096,'long4096':10240,'full50k':10240}[category])
    return {'tier':category,'rank':TIERS.index(category)+1,'required_TRAIN':required_train,
        'required_GEN':required_gen,'observed_pairs':sorted(pairs),'missing_pairs':sorted(required-pairs),
        'complete_registered_seed_grid':required.issubset(pairs),'terminal_budget':terminal,
        'complete_terminal_evidence':terminal and required.issubset(pairs),
        'full50k_membership_verified':category=='full50k' and all(r['native_membership']['full_membership'] and r['native_membership']['require_full_set'] for r in values)}

def build_data(main_art=MAIN_ART,main_runs=MAIN_RUNS,compact_art=ART,compact_runs=RUNS):
    collections=[collect('main',main_art,main_runs),collect('compact',compact_art,compact_runs)]
    all_rows=[row for collection in collections for row in collection['rows']]
    main_protocol=collections[0]['protocol'];compact_protocol=collections[1]['protocol']
    aligned=None
    if main_protocol and compact_protocol:
        aligned=all(compact_protocol['sources'].get(k)==v for k,v in main_protocol['sources'].items())
    for row in all_rows:row['cross_source_alignment']=aligned is True
    aliases=[];conflicts=[];unique={};blocked=set()
    for row in sorted(all_rows,key=lambda r:r.get('finished_utc','')):
        key=(row['source'],row['variant'],row['tier'],row['budget'],scope_key(row))
        if key in blocked:continue
        if key in unique:
            old=unique[key]
            conflicting=any(old.get(field)!=row.get(field) for field in METRICS+PAIR_FIELDS+('generated_onroad_fraction','num_generated_vehicles','num_gt_vehicles','checkpoint','generator_num_parameters','generator_trainable_parameters','frozen_map_num_parameters','initial_generator_sha256','native_metric_identity'))
            if conflicting:
                conflicts.append({'source':row['source'],'variant':row['variant'],'tier':row['tier'],
                    'scope':list(scope_key(row)),'stages':[old['stage'],row['stage']],
                    'policy':'Conflicting exact-scope repeats excluded from primary scoring; not averaged or silently overwritten'})
                del unique[key];blocked.add(key);continue
            aliases.append({'source':row['source'],'stage':row['stage'],'scope':list(scope_key(row)),
                'independent_replicate':False,'policy':'Identical verified exact-scope alias counted once'})
        unique[key]=row
    all_rows=list(unique.values())
    grouped=defaultdict(list)
    for row in all_rows:grouped[(row['source'],row['variant'],row['tier'],row['budget'],row['num_samples'],row['eval_manifest_sha256'])].append(row)
    registered=maybe(compact_art/'effective_plan.json',{})
    anchor_name=registered.get('selected_graph_anchor','norm_time_every_layer')
    summaries=[]
    for (source,name,category,budget,scenes,manifest),values in grouped.items():
        controls=primary_controls(values,all_rows);primary=comparison(values,controls)
        anchors=[r for r in all_rows if r['variant']==anchor_name]
        if source=='main':anchors=[r for r in anchors if r['source']=='main']
        anchor=comparison(values,matching(values,anchors))
        per_train=[]
        for seed in sorted({r['seed'] for r in values}):
            subset=[r for r in values if r['seed']==seed]
            paired=comparison(subset,primary_controls(subset,all_rows))
            reference=comparison(subset,matching(subset,anchors))
            per_train.append({'TRAIN':seed,'GEN':sorted({r['eval_seed'] for r in subset}),
                'native_metrics':mean_metrics(subset),'primary_fresh657':paired,'matched_initializer':reference,
                'same_fresh657_initializer':same_fresh657_initializer(paired,reference),
                'R_lt1_and_gates':paired['R']<1 and paired['all_TRAIN_gates_pass'] if paired['available'] else None})
        completeness=evidence_completeness(values)
        each_train=all(r['R_lt1_and_gates'] is True for r in per_train)
        validated_both=(completeness['complete_terminal_evidence'] and category in ('long4096','full50k'))
        resources={'actual_generator_parameters':sorted({r['generator_num_parameters'] for r in values if r.get('generator_num_parameters') is not None}),
            'actual_generator_trainable_parameters':sorted({r['generator_trainable_parameters'] for r in values if r.get('generator_trainable_parameters') is not None}),
            'actual_frozen_map_parameters':sorted({r['frozen_map_num_parameters'] for r in values if r.get('frozen_map_num_parameters') is not None})}
        summaries.append({'source':source,'variant':name,'tier':category,'budget':budget,'num_samples':scenes,
            'seen_training_examples':sorted({r['seen_training_examples'] for r in values}),
            'equivalent_epochs':statistics.fmean(r['seen_training_examples'] for r in values)/493542,
            'TRAIN':sorted({r['seed'] for r in values}),'GEN':sorted({r['eval_seed'] for r in values}),
            'eval_manifest_sha256':manifest,'evidence':completeness,'primary_fresh657':primary,
            'matched_initializer':anchor,'initializer_name':anchor_name,'per_TRAIN':per_train,
            'same_fresh657_initializer':same_fresh657_initializer(primary,anchor),
            'each_observed_TRAIN_R_lt1_and_gates':each_train,
            'both_TRAIN_terminal_improves_fresh657':each_train and primary.get('R',math.inf)<1 and primary.get('all_TRAIN_gates_pass',False) if validated_both else None,
            'native_mean_metrics':mean_metrics(values),'resources':resources,
            'complete_stages':sorted({r['stage'] for r in values}),'native_observations':values})
    summaries.sort(key=lambda r:(-TIERS.index(r['tier']),-r['budget'],r['source'],r['variant']))
    sources={collection['source']:{key:value for key,value in collection.items() if key!='rows'} for collection in collections}
    compact_complete=bool(collections[1]['verified_stages'])
    return {'generated_utc':datetime.now(timezone.utc).isoformat(),'report_kind':'combined_verified_main_and_compact',
        'compact_state':'VERIFIED_RESULTS_AVAILABLE' if compact_complete else 'WAITING_NO_COMPLETED_COMPACT_RESULTS',
        'sources':sources,'comparisons':summaries,'aliases_not_replicates':aliases,'duplicate_scope_conflicts':conflicts,
        'primary_policy':'Only outer/native/hash-verified fresh main657 or registered compact657; no historical denominator. Exact TRAIN/GEN/examples/manifest/data-order/calibration/map/EMA scope.',
        'shared_main_compact_training_sources_match':aligned,
        'metric_scales':dict(zip(METRICS,('JSD×10','JSD×10','JSD×100','JSD×100','JSD×100','JSD×100','generated-vehicle collision percent'))),
        'R_definition':'Mean of seven native candidate/control metric ratios; GEN mean within TRAIN, then equal TRAIN mean. R_attr means Length/Width/Speed ratios.',
        'gates_definition':'Length/Width/Speed ratios ≤1.15; on-road drop ≤.02; generated vehicle/GT ∈[.9,1.1]. Report pooled and every TRAIN separately; each TRAIN R<1 is needed for both-TRAIN improvement.',
        'comparison_denominator_definition':'B=candidate exact fresh657; A=matched initializer. With complete matched A, compact medium/long compares R(C/B)<min(1,R(A/B)) and gates against B, with additional phase/slope/per-TRAIN requirements. Direct R(C/A) and A gates are descriptive and not mathematically equivalent. Missing A is not estimated; long can use threshold 1 under its registered rule. Full50k confirmation compares against fresh657 only.',
        'architecture_rng_scope':compact_driver.ARCHITECTURE_RNG_SCOPE,
        'native_128epoch_full_evaluation_verified':False,
        'warning':'Screen/medium and10240-step (~10.62epochs) results are evidence at their actual budget only; they do not prove128epoch convergence. Missing/partial stages and unpaired denominators are excluded.',
        'decisions':{name:maybe(compact_art/name) for name in ('medium_selection.json','medium.json','long_selection.json','long.json','full.json') if (compact_art/name).is_file()}}


def fmt(value,digits=4):return '—' if value is None else f'{value:.{digits}f}'
def table(headers,rows):
    cells=lambda values:'<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    return '<div class="scroll"><table><thead>'+cells(headers)+'</thead><tbody>'+(''.join(cells(row) for row in rows) or '<tr><td colspan="'+str(len(headers))+'">等待完整核验的结果。</td></tr>')+'</tbody></table></div>'
def gate(result):return 'PASS' if result.get('all_TRAIN_gates_pass') else 'FAIL' if result.get('available') else '未配对'

def render_html(data,out):
    sections=[];per_train=[];metrics=[];resources=[]
    labels={'full50k':'完整50k评价','long4096':'10k steps /4096场景确认','medium':'4096 steps /4096场景中期','screen':'768 steps /1024场景筛选'}
    for category in reversed(TIERS):
        body=[]
        for row in data['comparisons']:
            if row['tier']!=category:continue
            primary=row['primary_fresh657'];anchor=row['matched_initializer'];evidence=row['evidence']
            common=row.get('same_fresh657_initializer') or same_fresh657_initializer(primary,anchor)
            complete='完整seed grid' if evidence['complete_terminal_evidence'] else '中间预算/seed确认不足'
            body.append([row['source'],row['variant'],row['budget'],fmt(row['equivalent_epochs'],3),row['num_samples'],','.join(map(str,row['TRAIN'])),','.join(map(str,row['GEN'])),fmt(primary.get('R')),fmt(primary.get('R_attr')),gate(primary),fmt(common.get('initializer_R')),fmt(anchor.get('R')),gate(anchor),complete,row['both_TRAIN_terminal_improves_fresh657']])
            for value in row['per_TRAIN']:
                p=value['primary_fresh657'];a=value['matched_initializer'];ratios=p.get('ratios',{})
                common=value.get('same_fresh657_initializer') or same_fresh657_initializer(p,a)
                per_train.append([category,row['source'],row['variant'],row['budget'],value['TRAIN'],','.join(map(str,value['GEN'])),fmt(p.get('R')),fmt(p.get('R_attr')),gate(p),value['R_lt1_and_gates'],fmt(common.get('initializer_R')),fmt(a.get('R')),gate(a)]+[fmt(ratios.get(k)) for k in METRICS])
            metrics.append([category,row['source'],row['variant'],row['budget'],row['num_samples']]+[fmt(row['native_mean_metrics'][k],6) for k in METRICS])
        sections.append('<h2>'+labels[category]+'</h2>'+table(['来源','候选','steps','等价epoch','场景','TRAIN','GEN','R(C/B)↓','R_attr/B↓','fresh657 B gates','R(A/B)↓','直接 R(C/A)↓','直接 A gates','证据完整度','两TRAIN均改善'],body))
    for source in data['sources'].values():
        for stage in source['verified_stages']:
            rows=[r for item in data['comparisons'] if item['source']==stage['source'] for r in item['native_observations'] if r['stage']==stage['stage']]
            counts=stage['actual_generator_parameters']
            r=stage['runtime']
            trainable=stage['actual_generator_trainable_parameters'];frozen=stage['actual_frozen_map_parameters']
            resources.append([stage['source'],stage['stage'],', '.join(f'{n:,}' for n in counts) or '未记录',', '.join(f'{n:,}' for n in trainable) or '未记录',', '.join(f'{n:,}' for n in frozen) or '未记录',fmt(stage['driver_elapsed_seconds'],1),fmt(r['train_seconds'],1),fmt(r['train_seconds_per_step'],3),fmt(r['eval_seconds'],1),fmt(r['cuda_peak_allocated_mib'],0),fmt(r['cuda_peak_reserved_mib'],0),fmt(r['peak_process_rss_mib'],0)])
    status='；'.join(source+': '+str(value['status'].get('status')) for source,value in data['sources'].items())
    waiting='<p class="notice">Compact 尚无完整核验结果：当前等待 main 队列和 GPU 空闲。下方 main 结果是已有证据，不代表 compact 已改善。</p>' if data['compact_state']=='WAITING_NO_COMPLETED_COMPACT_RESULTS' else ''
    failures=[item for source in data['sources'].values() for key in ('partials_not_evidence','quarantined') for item in source[key]]+data['duplicate_scope_conflicts']
    failtext=''.join('<li>'+html.escape(json.dumps(item,ensure_ascii=False))+'</li>' for item in failures) or '<li>暂无未完成或隔离记录。</li>'
    provenance=[]
    for name,source in data['sources'].items():
        provenance.append([name,source['protocol_sha256'] or '尚未冻结adaptive plan',source['source_bundle_sha256'] or '尚无执行source bundle',len(source['verified_stages'])])
    output=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>657：main＋compact已核验报告</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1550px;margin:24px auto;padding:0 20px;color:#18334b;background:#f6f9fc}}h1{{font-size:26px}}h2{{font-size:21px;margin-top:30px}}.notice{{background:#e8f1fa;padding:15px;border-left:4px solid #367aac}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px;background:white}}td{{padding:7px;border:1px solid #d4e0ea;text-align:right;white-space:nowrap}}thead td{{background:#e4eef7;font-weight:bold}}td:nth-child(-n+3){{text-align:left}}li,code{{overflow-wrap:anywhere}}@media print{{body{{margin:0;background:white}}table{{font-size:9px}}.scroll{{overflow:visible}}}}</style>
<h1>657 baseline：main＋compact 的分级配对证据</h1>
<p class="notice">状态：{html.escape(status)}。UTC 更新：{html.escape(data['generated_utc'])}。只读已成功退出并核验 source、summary、contract 与原生 metric SHA 的完整 stage。</p>{waiting}
<p>固定采样 N20、EMA、32-true；固定 AIRL map encoder，fresh G1。R 是七项 candidate/新657指标比的均值，先 GEN 在各 TRAIN 内均值，再 TRAIN 等权。R_attr 为 Length、Width、Speed 的指标比均值。没有相同 TRAIN、GEN、训练样本曝光及场景清单的完整新参考时显示“—”；禁止历史参考替代主分母。control 对自身 R=1。</p>
<p><b>分母说明：C 为候选，B 为该候选的配对 fresh657，A 为同范围 initializer。</b> 完整 A 可用时，Compact 的 medium/long 晋级 R 条件是 R(C/B)&lt;min(1,R(A/B))，门槛针对 B；还需满足阶段、曲线、参数和逐 TRAIN/GEN 的要求。R(A/B) 与 R(C/B) 使用同一个 B。直接 R(C/A) 和“直接 A gates”仅作诊断，并非晋级规则；这两个判断不数学等价。缺失 A 时不补值；long 按已冻结规则可使用阈值 1。full50k 确认只要求对 fresh657 改善，不使用 A 的 R 阈值。</p>
<p><b>两 TRAIN 改善须每个 TRAIN 各自 R&lt;1 且 gates 通过，并且 pooled 通过。</b> 单个种子、较短预算或不完整 stage 不能升级为更高证据。完整50k与4096/1024场景分开；10240 steps 约10.62个等价epoch，不证明128epoch最终收敛。</p>
{''.join(sections)}
<h2>每个 TRAIN 的 R、gates 与七项比值</h2>
<p>B gates：Length/Width/Speed 各 candidate/fresh657 比值≤1.15；onroad相对 B 最多下降.02；生成vehicle/GT∈[.9,1.1]。直接 A gates 用相同门槛改为相对 initializer。JSON保留每项门槛与分母路径；pooled R 不能隐藏某个 TRAIN 变差。</p>
{table(['等级','来源','候选','steps','TRAIN','GEN','R(C/B)↓','R_attr/B↓','fresh657 B gates','R<1且B gates','R(A/B)↓','直接 R(C/A)↓','直接 A gates']+[label+'比/B' for label in LABELS],per_train)}
<h2>七项原生指标</h2><p>Near、Lat 是JSD×10；Ang、Length、Width、Speed 是JSD×100；Collision 是生成vehicle的碰撞百分比。这里只汇总同一预算和场景范围，JSON保留各TRAIN/GEN原始值与实际manifest SHA。</p>
{table(['等级','来源','候选','steps','场景']+list(LABELS),metrics)}
<h2>实测参数、时间和内存</h2><p>参数仅来自实际 runner summary，计划数不会填充实测空值。时间为完整stage；CUDA峰值来自训练回调的进程累计训练峰值，不是评价或每层精确内存。</p>
{table(['来源','完整stage','实测G1总参数','实测G1可训参数','实测固定map参数','wall秒','训练秒','秒/step','评价秒','CUDA allocated MiB','CUDA reserved MiB','RSS MiB'],resources)}
<h2>配对范围与随机数限制</h2><p>架构变化共享 TRAIN seed/data order、曝光、GEN seed 和场景清单。seed_everything 在 G1 构造前；fresh training 没有在构造后重置 RNG。初始化随机数消耗随 width/depth/MLP/sincos 改变，逐step同一训练噪声轨迹未记录或保证。两TRAIN确认仍必要。图半径/K不增加参数；已完成 main screen 图臂的 initial G1 SHA 相同。</p>
<h2>来源 SHA</h2>{table(['来源','protocol SHA256','source bundle SHA256','完整stages'],provenance)}
<p>完整 source 文件SHA、每项原生报告/summary/contract SHA、checkpoint、scope、实际参数、每TRAIN/GEN七指标均在 <a href="live_report.json">live_report.json</a>。主/compact共享训练源一致性：{html.escape(str(data['shared_main_compact_training_sources_match']))}（None表示compact尚未冻结）。</p>
<h2>运行中、失败与隔离记录</h2><ul>{failtext}</ul>
<p>未完成或核验失败的 stage 不进入结果、排名或改善结论。lane_agentmax05 同时改变地图token条件与碰撞聚合/权重，是联合tradeoff假设，不能把结果单独归因于某一项。</p></html>'''
    atomic_text(out/'live_report.html',output)

def build(out=ART,**kwargs):
    data=build_data(**kwargs);write_report_json(out/'live_report.json',data);render_html(data,out);return data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ART)
    parser.add_argument('--watch',type=float,default=0,help='CPU-only refresh seconds; zero writes once')
    args=parser.parse_args()
    while True:
        data=build(args.out)
        print(json.dumps({'report':str(args.out/'live_report.html'),'compact_state':data['compact_state'],
            'verified_main_stages':len(data['sources']['main']['verified_stages']),
            'verified_compact_stages':len(data['sources']['compact']['verified_stages'])}),flush=True)
        if not args.watch:break
        time.sleep(max(5.,args.watch))

if __name__=='__main__':main()
