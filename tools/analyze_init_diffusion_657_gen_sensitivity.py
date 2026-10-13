"""CPU-only descriptive GEN sensitivity for verified terminal research cohorts.

No primary selection policy is changed. Two generation seeds are observations,
not enough for a strong confidence interval or final128epoch conclusion.
"""
from __future__ import annotations
import argparse, copy, hashlib, html, json, math, os, statistics, sys, tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import report_init_diffusion_657_compact as report
METRICS=report.METRICS
TIERS=('medium','long4096','full50k')


def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):d.update(chunk)
    return d.hexdigest()

def scope(row):
    return (row['seed'],row['eval_seed'],row['budget'],row['seen_training_examples'],
            row['tier'],row['eval_manifest_sha256'],row['num_samples'])

def same_scope(row,base):
    return (scope(row)==scope(base) and all(row[k]==base[k] for k in report.PAIR_FIELDS)
        and row['native_metric_identity']==base['native_metric_identity']
        and (row['source']==base['source'] or row.get('cross_source_alignment') is True))

def verify_native(row):
    path=Path(row['metrics_path']);require(sha(path)==row['metrics_sha256'],'Native metrics SHA changed: '+str(path))
    native=json.loads(path.read_text())
    require(all(math.isfinite(row[k]) and row[k]==native['agent_metrics'][k] for k in METRICS),'Native seven metrics differ')
    for key in ('generated_onroad_fraction','num_generated_vehicles','num_gt_vehicles'):
        require(row[key]==native[key],'Native gate/population field differs: '+key)
    require(native['num_samples']==native['num_gt_samples']==row['num_samples'],'Native scene count differs')
    require(native['runtime']['torch_initial_seed']==row['eval_seed'],'Native GEN seed differs')
    require(native['decoder']['ema_num_updates']==row['budget'] and native['decoder']['sampling_steps']==20,'Native budget/N20 differs')
    require(native['decoder']['use_ema'] is True and row['evaluation_weights']=='ema','Native weights differ')
    require(native['unverified_generation_timestep_count']==0,'Native generation membership unverified')
    if row['tier']=='full50k':require(native['full_membership'] is True and native['require_full_set'] is True,'Full50k membership absent')
    return {'path':str(path),'sha256':row['metrics_sha256']}

def denominator(row,declared,all_rows,variant):
    cells=[d for d in declared.get('denominator_cells',[]) if d['TRAIN']==row['seed'] and d['GEN']==row['eval_seed']]
    require(len(cells)==1,'Missing/duplicate declared same-TRAIN/GEN denominator')
    cell=cells[0]
    matches=[r for r in all_rows if r['source']==cell['source'] and r['stage']==cell['stage']
        and r['variant']==variant and r['metrics_path']==cell['metrics_path']
        and r['metrics_sha256']==cell['metrics_sha256'] and r['checkpoint']==cell['checkpoint']]
    require(len(matches)==1,'Declared fresh denominator is not a unique verified native observation')
    base=matches[0];require(same_scope(row,base),'Denominator differs in source/seed/budget/exposure/manifest/scene/data/map/EMA/native scope')
    require(row['source']=='compact' or base['source']=='main','Main candidate cannot fall back to compact/historical control')
    return base

def spread(values):
    mean=statistics.fmean(values);lo=min(values);hi=max(values)
    return {'n':len(values),'min':lo,'max':hi,'range':hi-lo,'mean':mean,'relative_range_over_mean':(hi-lo)/abs(mean) if mean else None}

def summarize_seed(cells,policy,anchor_policy):
    flags=[c['primary_fresh657']['R']<1 for c in cells]
    passing=[c['primary_fresh657']['R']<1 and c['primary_fresh657']['gates_pass'] for c in cells]
    gates={k:[c['primary_fresh657']['gates'][k] for c in cells] for k in cells[0]['primary_fresh657']['gates']}
    primary=report.comparison([c['candidate_observation'] for c in cells],[c['control_observation'] for c in cells])
    require(primary['available'] and math.isclose(primary['R'],policy['R'],rel_tol=1e-12),'Mean-GEN primary policy differs from recorded TRAIN result')
    return {'TRAIN':cells[0]['TRAIN'],'GEN':[c['GEN'] for c in cells],'primary_mean_GEN_policy':copy.deepcopy(policy),
        'matched_initializer_mean_GEN_policy':copy.deepcopy(anchor_policy),'cell_R':spread([c['primary_fresh657']['R'] for c in cells]),
        'cell_R_attr':spread([c['primary_fresh657']['R_attr'] for c in cells]),
        'cell_metric_ratios':{k:spread([c['primary_fresh657']['ratios'][k] for c in cells]) for k in METRICS},
        'observed_GEN_metric_improvement_count':{k:sum(c['primary_fresh657']['ratios'][k]<1 for c in cells) for k in METRICS},
        'GEN_metric_improvement_sign_disagreement':{k:len({c['primary_fresh657']['ratios'][k]<1 for c in cells})>1 for k in METRICS},
        'candidate_native_metric_spread':{k:spread([c['candidate_observation'][k] for c in cells]) for k in METRICS},
        'control_native_metric_spread':{k:spread([c['control_observation'][k] for c in cells]) for k in METRICS},
        'observed_GEN_R_lt1_count':sum(flags),'observed_GEN_R_lt1_and_gates_count':sum(passing),
        'each_GEN_R_lt1':all(flags),'GEN_R_sign_disagreement':any(flags) and not all(flags),
        'each_GEN_R_lt1_and_gates':all(passing),'GEN_gates_disagree':{k:len(set(v))>1 for k,v in gates.items()},
        'mean_GEN_policy_improves':policy['R']<1 and policy['gates_pass'],
        'mean_policy_improves_but_only_one_GEN_R_lt1':policy['R']<1 and policy['gates_pass'] and sum(flags)==1,
        'mean_policy_improves_but_only_one_GEN_R_lt1_and_gates':policy['R']<1 and policy['gates_pass'] and sum(passing)==1,
        'descriptive_cell_R_mean':statistics.fmean(c['primary_fresh657']['R'] for c in cells),
        'descriptive_mean_of_cell_R_is_primary_R':False,
        'note':'Cell R range is descriptive only. Primary R averages native metrics over GEN within TRAIN before forming ratios.'}

def analyze(data, *, native_verifier=verify_native):
    require(data.get('report_kind')=='combined_verified_main_and_compact','Expected combined verified report')
    all_rows=[r for group in data['comparisons'] for r in group['native_observations']]
    cohorts=[];excluded=[];verified_paths={}
    for group in data['comparisons']:
        if group['tier'] not in TIERS or not group['evidence'].get('complete_terminal_evidence'):continue
        identity=group['source']+':'+group['variant']
        try:
            require(json.loads(json.dumps(report.evidence_completeness(group['native_observations'])))==group['evidence'],'Terminal seed-grid declaration differs')
            require(group['primary_fresh657'].get('available'),'No exact fresh primary denominator')
            cells=[]
            for row in group['native_observations']:
                base=denominator(row,group['primary_fresh657'],all_rows,'control657')
                for r in (row,base):
                    if r['metrics_path'] not in verified_paths:verified_paths[r['metrics_path']]=native_verifier(r)
                result=report.comparison([row],[base]);require(result['available'],'Cell fresh comparison failed')
                anchor=None
                if group['matched_initializer'].get('available'):
                    anchor_row=denominator(row,group['matched_initializer'],all_rows,group['initializer_name'])
                    if anchor_row['metrics_path'] not in verified_paths:verified_paths[anchor_row['metrics_path']]=native_verifier(anchor_row)
                    anchor=report.comparison([row],[anchor_row]);require(anchor['available'],'Cell initializer comparison failed')
                cells.append({'TRAIN':row['seed'],'GEN':row['eval_seed'],'source_pair':[row['source'],base['source']],
                    'primary_fresh657':result,'matched_initializer':anchor or {'available':False},
                    'candidate_observation':row,'control_observation':base})
            policy=report.comparison([c['candidate_observation'] for c in cells],[c['control_observation'] for c in cells])
            require(policy==group['primary_fresh657'],'Recomputed full primary policy differs from report')
            by_train=[]
            for train in group['TRAIN']:
                subset=sorted([c for c in cells if c['TRAIN']==train],key=lambda c:c['GEN'])
                per=[v for v in group['per_TRAIN'] if v['TRAIN']==train];require(len(per)==1,'Missing recorded TRAIN policy')
                by_train.append(summarize_seed(subset,per[0]['primary_fresh657'],per[0]['matched_initializer']))
            cohorts.append({'identity':identity,'source':group['source'],'variant':group['variant'],'tier':group['tier'],
                'budget':group['budget'],'num_samples':group['num_samples'],'TRAIN':group['TRAIN'],'GEN':group['GEN'],
                'eval_manifest_sha256':group['eval_manifest_sha256'],'evidence':group['evidence'],
                'primary_mean_GEN_within_TRAIN_policy':group['primary_fresh657'],'matched_initializer_policy':group['matched_initializer'],
                'initializer_name':group['initializer_name'],'per_TRAIN':by_train,'cells':cells})
        except (ValueError,RuntimeError,KeyError,ZeroDivisionError,OSError) as error:
            excluded.append({'identity':identity,'tier':group['tier'],'budget':group['budget'],'reason':str(error)})
    rankings=cohort_rankings(cohorts)
    return {'generated_utc':datetime.now(timezone.utc).isoformat(),'analysis_kind':'verified_descriptive_GEN_sensitivity',
        'cohorts':cohorts,'rankings_within_exact_cohort_only':rankings,'excluded_not_evidence':excluded,
        'verified_native_metrics':list(verified_paths.values()),'main_policy_unchanged':True,
        'primary_policy':data['R_definition'],'gates_definition':data['gates_definition'],
        'limitations':['Only complete terminal4096-step medium and complete registered long/full seed grids are included.',
            'Per-GEN results are descriptive. Two GEN observations cannot justify strong confidence intervals.',
            'Primary policy remains mean native metrics over GEN within TRAIN, then equal TRAIN mean, then seven ratios.',
            'No historical or unregistered cross-source/cohort fallback. Each denominator is the declared hash-verified fresh cell.',
            'GEN variation is conditional on each trained checkpoint; it is not independent TRAIN replication or128epoch evidence.',
            'Direct candidate/initializer R uses seven actual metric ratios; it is not candidate-vs657 R divided by initializer-vs657 R.'],
        'Torch_imported':'torch' in sys.modules,'GPU_started':False}

def cohort_rankings(cohorts):
    grouped=defaultdict(list)
    for c in cohorts:
        if c['variant']=='control657':continue
        denom=tuple(sorted((cell['TRAIN'],cell['GEN'],cell['control_observation']['source'],cell['control_observation']['checkpoint'],cell['control_observation']['metrics_sha256']) for cell in c['cells']))
        key=(c['source'],c['tier'],c['budget'],c['num_samples'],c['eval_manifest_sha256'],tuple(c['TRAIN']),tuple(c['GEN']),denom)
        grouped[key].append(c)
    result=[]
    for key,values in grouped.items():
        by_gen=[]
        for train in key[5]:
            for gen in key[6]:
                rows=[{'identity':c['identity'],'R':cell['primary_fresh657']['R'],'gates_pass':cell['primary_fresh657']['gates_pass']}
                    for c in values for cell in c['cells'] if cell['TRAIN']==train and cell['GEN']==gen]
                rows.sort(key=lambda v:v['R']);passed=[v for v in rows if v['gates_pass']]
                by_gen.append({'TRAIN':train,'GEN':gen,'descriptive_order_by_cell_R':rows,'lowest_cell_R_with_gates':passed[0]['identity'] if passed else None})
        policy=sorted([{'identity':v['identity'],'R':v['primary_mean_GEN_within_TRAIN_policy']['R'],
            'gates_pass':v['primary_mean_GEN_within_TRAIN_policy']['all_TRAIN_gates_pass']} for v in values],key=lambda v:v['R'])
        train_rankings=[]
        for train in key[5]:
            orders=sorted([{'identity':c['identity'],'R':v['primary_mean_GEN_policy']['R'],'gates_pass':v['primary_mean_GEN_policy']['gates_pass']}
                for c in values for v in c['per_TRAIN'] if v['TRAIN']==train],key=lambda v:v['R'])
            passing=[v for v in orders if v['gates_pass']]
            within=[v['lowest_cell_R_with_gates'] for v in by_gen if v['TRAIN']==train]
            train_rankings.append({'TRAIN':train,'mean_GEN_order':orders,'lowest_mean_GEN_R_with_gates':passing[0]['identity'] if passing else None,
                'GEN_best_passing_candidate_differs_within_TRAIN':len(set(within))>1})
        result.append({'source':key[0],'tier':key[1],'budget':key[2],'num_samples':key[3],'eval_manifest_sha256':key[4],
            'TRAIN':list(key[5]),'GEN':list(key[6]),'mean_GEN_primary_order':policy,'per_GEN_descriptive_rankings':by_gen,
            'per_TRAIN_mean_GEN_rankings':train_rankings,
            'GEN_best_passing_candidate_differs':any(v['GEN_best_passing_candidate_differs_within_TRAIN'] for v in train_rankings),
            'TRAIN_best_passing_candidate_differs':len({v['lowest_mean_GEN_R_with_gates'] for v in train_rankings})>1})
    return result

def verified_snapshot(path, *, source_bytes=None):
    raw=Path(path).read_bytes() if source_bytes is None else source_bytes;data=json.loads(raw);sources=data['sources']
    rebuilt=json.loads(json.dumps(report.build_data(main_art=Path(sources['main']['artifact_dir']),main_runs=Path(sources['main']['run_dir']),
        compact_art=Path(sources['compact']['artifact_dir']),compact_runs=Path(sources['compact']['run_dir']))))
    expected={(g['source'],g['variant'],g['tier'],g['budget'],g['eval_manifest_sha256']):g for g in rebuilt['comparisons']}
    accepted_keys=set();excluded=[]
    for group in data['comparisons']:
        key=(group['source'],group['variant'],group['tier'],group['budget'],group['eval_manifest_sha256'])
        if group['tier'] not in TIERS or not group['evidence'].get('complete_terminal_evidence'):continue
        actual=expected.get(key)
        if actual is None or actual!=group:
            excluded.append({'identity':group['source']+':'+group['variant'],'tier':group['tier'],'budget':group['budget'],'reason':'Report cohort no longer matches current hash-verified native stage evidence; refresh combined report'})
        else:accepted_keys.add(key)
    # Keep all verified rows for declared fresh/anchor denominator lookup, while
    # permitting sensitivity analysis only for the snapshot's complete cohorts.
    data=copy.deepcopy(rebuilt);data['comparisons']=[v for v in rebuilt['comparisons'] if (v['source'],v['variant'],v['tier'],v['budget'],v['eval_manifest_sha256']) in accepted_keys or not v['evidence'].get('complete_terminal_evidence') or v['tier'] not in TIERS]
    return data,{'path':str(Path(path).resolve()),'sha256':hashlib.sha256(raw).hexdigest()},excluded

def render_html(data):
    esc=lambda v:html.escape(str(v));fmt=lambda v:f'{v:.5f}'
    parts=['<!doctype html><meta charset="utf-8"><title>GEN sensitivity</title><style>body{font:15px system-ui;margin:28px;max-width:1600px}table{border-collapse:collapse;margin:12px 0 28px;font-size:13px}td,th{border:1px solid #ccd;padding:7px}th{background:#eef}p{max-width:1100px}.fail{color:#b22}</style>',
        '<h1>采样种子敏感性：已验证的完整终点</h1><p>主政策不变：先在每个 TRAIN 内平均 GEN 的原生指标，再计算 R。单 GEN 用于观察，不用于替代主选择。两个 GEN 不足以支持强置信区间。</p>',
        '<p>Report SHA: '+esc(data['source_report']['sha256'])+' · '+esc(data['generated_utc'])+'</p>']
    if not data['cohorts']:parts.append('<p>等待完整 terminal medium/long/full 证据；部分结果未纳入。</p>')
    for c in data['cohorts']:
        parts.append('<h2>'+esc(c['identity'])+' · '+esc(c['tier'])+' · step '+str(c['budget'])+' · N'+str(c['num_samples'])+'</h2>')
        parts.append('<p>主政策 R='+fmt(c['primary_mean_GEN_within_TRAIN_policy']['R'])+'；直接 initializer R='+ (fmt(c['matched_initializer_policy']['R']) if c['matched_initializer_policy']['available'] else 'unavailable')+'；实际 eval manifest '+esc(c['eval_manifest_sha256'])+'</p>')
        headers=['source/control','TRAIN','GEN','cell R','R_attr','gates','Near ratio','Lat ratio','Ang ratio','Length ratio','Width ratio','Speed ratio','Collision ratio','anchor cell R/gates']
        parts.append('<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>')
        for cell in c['cells']:
            score=cell['primary_fresh657'];anchor=cell['matched_initializer'];values=['/'.join(cell['source_pair']),cell['TRAIN'],cell['GEN'],fmt(score['R']),fmt(score['R_attr']),'PASS' if score['gates_pass'] else 'FAIL: '+','.join(k for k,v in score['gates'].items() if not v)]+[fmt(score['ratios'][k]) for k in METRICS]+[(fmt(anchor['R'])+' / '+('PASS' if anchor['gates_pass'] else 'FAIL: '+','.join(k for k,v in anchor['gates'].items() if not v))) if anchor['available'] else 'unavailable']
            parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
        parts.append('</table>')
        for train in c['per_TRAIN']:
            text=f"TRAIN {train['TRAIN']}：cell R范围 {fmt(train['cell_R']['min'])}–{fmt(train['cell_R']['max'])}；Speed ratio范围 {fmt(train['cell_metric_ratios']['speed_jsd']['min'])}–{fmt(train['cell_metric_ratios']['speed_jsd']['max'])}；各 GEN R<1 且gate通过：{train['each_GEN_R_lt1_and_gates']}。"
            parts.append('<p>'+esc(text)+'</p>')
    parts.append('<h2>证据限制</h2><ul>'+''.join('<li>'+esc(v)+'</li>' for v in data['limitations'])+'</ul>')
    if data['excluded_not_evidence']:parts.append('<pre>'+esc(json.dumps(data['excluded_not_evidence'],ensure_ascii=False,indent=2))+'</pre>')
    return '\n'.join(parts)

def freeze_input_snapshot(raw, directory):
    """Preserve exact input bytes with an atomic, non-overwriting SHA filename."""
    require(isinstance(raw,bytes),'Input snapshot requires the captured raw bytes')
    digest=hashlib.sha256(raw).hexdigest(); directory=Path(directory)
    path=directory/(digest+'.json')
    metadata={'path':str(path.resolve()),'sha256':digest,'size_bytes':len(raw),'immutable_exact_raw_input':True}
    directory.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==raw,'Existing content-addressed input snapshot differs: '+str(path))
        return metadata
    fd,temporary=tempfile.mkstemp(prefix=digest+'.',suffix='.tmp',dir=directory)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            require(path.read_bytes()==raw,'Concurrent content-addressed input snapshot differs: '+str(path))
        require(path.read_bytes()==raw,'Created content-addressed input snapshot differs: '+str(path))
    finally:Path(temporary).unlink(missing_ok=True)
    return metadata

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=report.ART/'live_report.json');parser.add_argument('--output-dir',type=Path,default=report.ART);args=parser.parse_args(argv)
    source_bytes=args.source.read_bytes();data,source,excluded=verified_snapshot(args.source,source_bytes=source_bytes);result=analyze(data);result['excluded_not_evidence']+=excluded
    result['tool']={'path':str(Path(__file__)),'sha256':sha(__file__)};result['source_bundle_sha256']={k:v['source_bundle_sha256'] for k,v in data['sources'].items()}
    require(not result['Torch_imported'],'CPU analysis unexpectedly imported Torch')
    snapshot=freeze_input_snapshot(source_bytes,args.output_dir/'gen_sensitivity_inputs')
    require(snapshot['sha256']==source['sha256'],'Captured input SHA differs from verified snapshot')
    result['source_report']={'path':snapshot['path'],'sha256':snapshot['sha256'],'origin_path':source['path'],'snapshot':snapshot}
    report.write_report_json(args.output_dir/'gen_sensitivity.json',result);report.atomic_text(args.output_dir/'gen_sensitivity.html',render_html(result))
    print(json.dumps({'cohorts':len(result['cohorts']),'excluded':result['excluded_not_evidence'],'json':str(args.output_dir/'gen_sensitivity.json'),'html':str(args.output_dir/'gen_sensitivity.html'),'Torch_imported':False},indent=2))
    return result

if __name__=='__main__':main()
