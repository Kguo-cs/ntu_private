"""Serialized adaptive frozen-map search; no pretrained generator is loaded.

The initial 14-arm stage must already be running. All later stages use the
repository scratch runner and metric implementation. Selection is recorded
before independent-seed confirmation; the full-set audit never reselects.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from tools.scratch_component_loss_sweep import SEARCH_CONFIG_KEYS

PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
ART = ROOT / 'docs/experiments/init_diffusion_frozen_map_search_20261011'
RUNS = ROOT / 'logs/component_loss_frozen_map'
DESIGN = ART / 'design'
MAP = ROOT / 'src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt'
LINEAR = ROOT / 'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'
LOG = ART / 'calibration_log_speed/normalizer.json'
METRICS = ('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
ATTR = ('length_jsd','width_jsd','speed_jsd')
SCALES = (1.,1.,10.,20.,15.,20.,5.)
PREFIX = 'model.model_config.decoder.init_diffusion.'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def equal_state(left, right):
    """Exact recursive equality for model/EMA/Adam/RNG provenance checks."""
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left,right)
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return type(left)==type(right) and len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def run(label, command):
    command = list(command)
    if not command or any(not isinstance(value, str) for value in command):
        raise ValueError('Stage command must contain exact string argv entries')
    if Path(label).name != label or label in ('', '.', '..'):
        raise ValueError('Stage label must be a safe basename')
    if command.count('--output') != 1 or command.index('--output') + 1 >= len(command):
        raise ValueError('Stage command must declare exactly one --output directory')
    output = Path(command[command.index('--output') + 1])
    output = (output if output.is_absolute() else ROOT / output).resolve()
    log_path = ROOT / ('logs/component_loss_frozen_map_'+label+'.log')
    manifest_path = ART / 'stage_executions' / (label+'.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def durable_manifest(value):
        temporary = manifest_path.with_suffix('.tmp')
        with temporary.open('w') as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
        descriptor = os.open(manifest_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def verify_completion():
        protocol_path, summary_path = output/'protocol.json', output/'summary.json'
        try:
            protocol, summary = read(protocol_path), read(summary_path)
            lines = log_path.read_text().splitlines()
        except (OSError, ValueError) as error:
            raise RuntimeError(f'{label}: completion artifacts unreadable; manual checkpoint recovery needed: {error}') from error
        if not isinstance(protocol, dict) or not protocol or not isinstance(summary, list) or not summary:
            raise RuntimeError(f'{label}: completion protocol/summary incomplete; manual checkpoint recovery needed')
        if any(not isinstance(row, dict) for row in summary) or not lines or lines[-1].strip() != 'DONE '+str(output):
            raise RuntimeError(f'{label}: terminal runner DONE does not verify this output; manual checkpoint recovery needed')
        return {name: {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                for name,path in (('protocol',protocol_path),('summary',summary_path),('log',log_path))}

    existing = read(manifest_path) if manifest_path.is_file() else None
    if existing is not None:
        if not isinstance(existing, dict) or existing.get('command') != command or existing.get('output') != str(output):
            raise RuntimeError(f'{label}: stage execution argv changed; refusing replay into an existing stage')
        if existing.get('returncode') != 0 or existing.get('status') not in ('completed', 'success'):
            raise RuntimeError(f'{label}: prior stage is partial/failed; manual checkpoint recovery needed, no automatic replay')
        artifacts = verify_completion()
        if existing.get('artifacts') is not None and existing['artifacts'] != artifacts:
            raise RuntimeError(f'{label}: completion artifacts changed; refusing to reuse an unverifiable stage')
        existing.update(status='success', artifacts=artifacts, last_verified_unix=time.time())
        durable_manifest(existing)
        save(ART / 'queue_active.json', {'stage':label,'command':command,'controller_pid':os.getpid(),
                                       'status':'SKIPPED_VERIFIED_COMPLETE'})
        print('SKIP VERIFIED COMPLETE '+label, flush=True)
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError(f'{label}: output exists without a verified execution manifest; manual checkpoint recovery needed')
    execution = {'stage':label,'command':command,'output':str(output),'log':str(log_path),
                 'controller_pid':os.getpid(),'started_unix':time.time(),'status':'running','returncode':None}
    durable_manifest(execution)
    print('START '+label, flush=True)
    save(ART / 'queue_active.json', {'stage':label,'command':command,'controller_pid':os.getpid()})
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            execution['child_pid'] = process.pid
            durable_manifest(execution)
            save(ART / 'queue_active.json', {'stage':label,'command':command,
                                           'controller_pid':os.getpid(),'child_pid':process.pid})
            result = process.wait()
    except (OSError, subprocess.SubprocessError) as error:
        execution.update(status='launch_failed', ended_unix=time.time(), error=str(error))
        durable_manifest(execution)
        raise RuntimeError(f'{label}: subprocess failed; manual checkpoint recovery needed: {error}') from error
    execution.update(status='failed' if result else 'completed', returncode=result, ended_unix=time.time())
    durable_manifest(execution)
    if result:
        raise RuntimeError(f'{label} failed with return code {result}; queue stopped; manual checkpoint recovery needed')
    execution.update(status='success', artifacts=verify_completion(), last_verified_unix=time.time())
    durable_manifest(execution)
    print('DONE '+label, flush=True)


def base(*, batch=512, scenes=1024, seed=817, normalizer=LINEAR, workers=0, confirmation=False):
    command = [PYTHON, 'tools/scratch_component_loss_sweep.py','--map-checkpoint',str(MAP),
               '--batch-size',str(batch),'--scenes',str(scenes),'--eval-batch-size','128',
               '--workers',str(workers),'--seed',str(seed),'--eval-seed','2817',
               '--normalizer-json',str(normalizer),'--evaluation-weights','ema']
    if workers:
        command += ['--sharing-strategy','file_system']
    if confirmation:
        command += ['--eval-manifest',str(ART/'confirmation4096_files.json'),'--extra-eval-seed','3817']
    return command


def endpoints(folder, budget):
    rows = [row for row in read(folder/'summary.json') if row['budget']==budget
            and row['evaluation_weights']=='ema' and row['eval_seed']==2817]
    if len({row['variant'] for row in rows}) != len(rows):
        raise ValueError('Duplicate endpoints')
    return {row['variant']:row for row in rows}


def compare(row, control):
    ratios = {key:row[key]/max(control[key],1e-6) for key in METRICS}
    eligible = (row['generated_onroad_fraction'] >= control['generated_onroad_fraction']-.02
                and .9 <= row['num_generated_vehicles']/row['num_gt_vehicles'] <=1.1
                and all(ratios[key]<=1.15 for key in ATTR))
    return dict(row, R=sum(ratios.values())/7, R_attr=sum(ratios[key] for key in ATTR)/3,
                S=sum(row[key]/scale for key,scale in zip(METRICS,SCALES))/7,
                relative_ratios=ratios, eligible=eligible)


def candidate(name, row):
    folder=Path(row['checkpoint']).parent
    cfg=OmegaConf.load(folder/'config.yaml')
    protocol=read(folder.parent/'protocol.json')
    return {'name':name,'recipe':OmegaConf.to_container(cfg.model.model_config.decoder.init_diffusion.component_loss,resolve=True),
            'config':{key:OmegaConf.select(cfg,key) for key in sorted(SEARCH_CONFIG_KEYS)
                      if key != 'model.model_config.lr'},
            'lr':float(cfg.model.model_config.lr),'clip':float(cfg.trainer.gradient_clip_val),
            'batch':int(cfg.data.train_batch_size),'normalizer':str(Path(protocol['normalizer_json']).resolve()),
            'checkpoint':row['checkpoint'],'source_row':row}


def fresh(label, arms, *, steps, checkpoints, scenes=1024, seed=817, workers=0, confirmation=False):
    # One runner invocation per shared normalizer/batch/recipe; all G1s fresh.
    recipe_file=DESIGN/(label+'_recipe.json'); variant_file=DESIGN/(label+'_variants.json')
    save(recipe_file,arms[0]['recipe'])
    variants={}
    for arm in arms:
        changes={'components':arm['recipe']['components'],
                 **{key:arm['recipe'][key] for key in ('heading_space','std_floor','normalize_time_weights')},
                 '__lr':arm['lr'],'__clip':arm['clip'],'__config':arm['config']}
        variants[arm['name']]=changes
    save(variant_file,variants)
    if len({(arm['batch'],arm['normalizer']) for arm in arms}) !=1:
        raise ValueError('Fresh group must share batch and normalizer')
    batch=arms[0]['batch']
    command=base(batch=batch,scenes=scenes,seed=seed,normalizer=Path(arms[0]['normalizer']),workers=workers,
                 confirmation=confirmation)+[
        '--steps',str(steps),'--checkpoint-steps',','.join(map(str,checkpoints)),
        '--lr',str(arms[0]['lr']),'--recipe',str(recipe_file),'--variant-file',str(variant_file),
        '--variants',','.join(variants),'--output',str(RUNS/label)]
    if steps*batch >=493542:
        command+=['--allow-multiple-epochs']
    run(label,command)
    return endpoints(RUNS/label,steps)


def aggregate(rows):
    by_train={}
    for row in rows:
        by_train.setdefault(row['seed'],[]).append(row)
    return {key:sum(sum(row[key] for row in group)/len(group) for group in by_train.values())/len(by_train)
            for key in METRICS+('generated_onroad_fraction','num_generated_vehicles','num_gt_vehicles')}


def main():
    os.chdir(ROOT)
    phase1=RUNS/'phase1'; phase1_log=ROOT/'logs/component_loss_frozen_map_phase1.log'
    while not (phase1_log.exists() and 'DONE '+str(phase1) in phase1_log.read_text()):
        # The runner prints its Path exactly as CLI supplied, which can be relative.
        if phase1_log.exists() and 'DONE logs/component_loss_frozen_map/phase1' in phase1_log.read_text():
            break
        if phase1_log.exists() and ('Traceback (most recent call last)' in phase1_log.read_text()
                                   or time.time()-phase1_log.stat().st_mtime>900):
            raise RuntimeError('Initial stage failed or stopped progressing; queue stopped')
        time.sleep(5)
    rows=endpoints(phase1,512)
    if len(rows)!=14:
        raise ValueError('Expected all14 complete initial arms')
    control=rows['new_control']
    ranked=sorted((compare(row,control) for row in rows.values()),key=lambda row:(not row['eligible'],row['R'],row['variant']))
    anchor_name=ranked[0]['variant']; anchor=candidate(anchor_name,rows[anchor_name])
    save(ART/'phase1_selection.json',{'control':control,'all_ranked':ranked,'anchor':anchor_name,
                                    'note':'Adaptive validation selection, not untouched test'})

    # Test worker speed and exact state equivalence before using it for later jobs.
    import torch
    benchmark=candidate('new_control',control)
    try:
        benchrows=fresh('worker2_benchmark',[benchmark],steps=128,checkpoints=[],scenes=64,workers=2)
        tested=torch.load(benchrows['new_control']['checkpoint'],map_location='cpu',weights_only=False)
        reference=torch.load(phase1/'new_control/step_128.ckpt',map_location='cpu',weights_only=False)
        mismatches=[key for key,value in reference['state_dict'].items() if torch.is_tensor(value)
                    and not torch.equal(value,tested['state_dict'][key])]
        elapsed=float(re.search(r'STEP 128:.*?elapsed=([0-9.]+)s',phase1_log.read_text()).group(1))
        worker_audit={'tensor_state_bitwise_equal':not mismatches,'mismatches':mismatches,
                      'new_fit_seconds':benchrows['new_control']['train_seconds'],
                      'reference_trace_elapsed_seconds':elapsed,
                      'excluded_from_main_model_and_metric_counts':True}
        er=reference['state_dict']['encoder.init_decoder._extra_state']['ema']
        et=tested['state_dict']['encoder.init_decoder._extra_state']['ema']
        worker_audit['ema_bitwise_equal']=len(er['shadow_params'])==len(et['shadow_params']) and all(
            torch.equal(a,b) for a,b in zip(er['shadow_params'],et['shadow_params']))
        worker_audit['ema_scalar_state_equal']=equal_state({k:v for k,v in er.items() if k!='shadow_params'},
                                                          {k:v for k,v in et.items() if k!='shadow_params'})
        worker_audit['adam_state_equal']=equal_state(reference['optimizer_states'],tested['optimizer_states'])
        worker_audit['training_rng_equal']=equal_state(reference['scratch_sweep_rng'],tested['scratch_sweep_rng'])
        worker_audit['timing_scope']='Single order-specific cold/warm pilot, fit timing vs first128 trace elapsed; not causal worker-speed evidence.'
        workers=2 if (not mismatches and worker_audit['ema_bitwise_equal'] and worker_audit['ema_scalar_state_equal']
                      and worker_audit['adam_state_equal'] and worker_audit['training_rng_equal']
                      and benchrows['new_control']['train_seconds']<elapsed) else 0
        del tested,reference
    except RuntimeError as error:
        worker_audit={'failed':True,'failure':str(error),'chosen_workers':0,
                      'excluded_from_main_model_and_metric_counts':True}
        workers=0
    worker_audit['chosen_workers']=workers
    save(ART/'actual_worker_benchmark.json',worker_audit)

    # Extend four loss arms within the first epoch, with true optimizer/RNG continuation.
    retain=['new_control','old_control']
    for row in ranked:
        if row['variant'] not in retain:
            retain.append(row['variant'])
        if len(retain)==4:
            break
    resume_file=DESIGN/'phase1_extend_resume.json';save(resume_file,{name:rows[name]['checkpoint'] for name in retain})
    run('phase1_extend',base(workers=workers)+['--steps','768','--checkpoint-steps','',
        '--recipe',str(DESIGN/'base_recipe.json'),'--variant-file',str(DESIGN/'phase1_variants.json'),
        '--variants',','.join(retain),'--resume-map',str(resume_file),'--output',str(RUNS/'phase1_extend')])
    extended=endpoints(RUNS/'phase1_extend',768); extcontrol=extended['new_control']
    # One recorded loss interaction, motivated by separately observed512-step
    # shape and speed gains. This is a fresh generator, not a checkpoint blend.
    joint_names=('shape_p2','speed4')
    joint_evidence=[compare(rows[name],control) for name in joint_names]
    joint_enabled=all(row['eligible'] and row['R']<.98 for row in joint_evidence)
    save(ART/'loss_interaction_protocol.json',{'single_factors':joint_evidence,
         'enabled':joint_enabled,'target_exposure':393216,'fresh_G1':True,
         'rule':'Test shape_p2+speed4 only if both512-step individual factors pass gates and R<0.98.'})
    if joint_enabled:
        joint=candidate('shape_p2_speed4',control)
        joint['recipe']['components']['length']['time_power']=2
        joint['recipe']['components']['width']['time_power']=2
        joint['recipe']['components']['velocity']['weight']=4.0
        extended.update(fresh('loss_interaction',[joint],steps=768,
                              checkpoints=[256,512],workers=workers))
    rankedext=sorted((compare(row,extcontrol) for row in extended.values()),key=lambda row:(not row['eligible'],row['R'],row['variant']))
    anchor=candidate(rankedext[0]['variant'],extended[rankedext[0]['variant']])
    save(ART/'phase2_anchor_selection.json',{'anchor':anchor,'ranking':rankedext})

    # Individual training/model interventions; same physical conditions and targets.
    parameter_specs={
        'lr2e4':{'lr':2e-4},'lr1e3':{'lr':1e-3},
        'clip2':{'clip':2.0},
        'optimizer_sd':{'config':{'model.model_config.optimizer':'scenario_dreamer','model.model_config.lr_warmup_steps':0}},
        'shape_uniform':{'config':{PREFIX+'shape_source':'uniform'}},
        'speed_uniform':{'config':{PREFIX+'velocity_source':'uniform'}},
        'edge_mlp':{'config':{PREFIX+'edge_embedding_type':'mlp'}},
        'ego_sincos':{'config':{PREFIX+'ego_context_heading_encoding':'sincos'}},
        'count_embedding':{'config':{PREFIX+'count_embedding_type':'scenario_dreamer',
                                    PREFIX+'count_max_num_agents':128,
                                    PREFIX+'count_max_num_lanes':4096}},
        'size_log':{'config':{PREFIX+'size_representation':'log'},'normalizer':str(LOG)},
        'batch256':{'batch':256},
    }
    save(ART/'phase2_parameter_protocol.json',{'anchor':anchor['name'],'parameters':parameter_specs,
         'target_training_exposure':393216,'primary_budget512':768,'budget256':1536,
         'all_from_fresh_G1':True,'normalizers_share_TRAIN8192_manifest':True})
    arms=[]
    for name,change in parameter_specs.items():
        arm=copy.deepcopy(anchor);arm['name']=name
        arm['config'].update(change.get('config',{}))
        for key in ('lr','normalizer','batch','clip'):
            if key in change:arm[key]=change[key]
        arms.append(arm)
    mainarms=[arm for arm in arms if arm['normalizer']==str(LINEAR) and arm['batch']==512]
    parameter_rows=fresh('phase2',mainarms,steps=768,checkpoints=[256,512],workers=workers)
    for arm in arms:
        if arm in mainarms:continue
        steps=1536 if arm['batch']==256 else 768
        parameter_rows.update(fresh('phase2_'+arm['name'],[arm],steps=steps,
                                   checkpoints=[steps//3,steps*2//3],workers=workers))
    allrows={**extended,**parameter_rows}
    compared=sorted((compare(row,extcontrol) for row in allrows.values()),key=lambda row:(not row['eligible'],row['R'],row['variant']))
    save(ART/'phase2_selection.json',{'reference':extcontrol,'ranked':compared})

    # At most two compatible combinations of individually demonstrated gains.
    beneficial=[row for row in compared if row['variant'] in parameter_specs and row['eligible'] and row['R']<compare(extended[anchor['name']],extcontrol)['R']*.98]
    combos=[]
    for i,left in enumerate(beneficial):
        for right in beneficial[i+1:]:
            a=parameter_specs[left['variant']];b=parameter_specs[right['variant']]
            keys_a=set(a.get('config',{}))|({key for key in a if key!='config'})
            keys_b=set(b.get('config',{}))|({key for key in b if key!='config'})
            if keys_a & keys_b:continue
            arm=copy.deepcopy(anchor);arm['name']='combo_'+left['variant']+'_'+right['variant']
            for spec in (a,b):
                arm['config'].update(spec.get('config',{}))
                for key in ('lr','normalizer','batch','clip'):
                    if key in spec:arm[key]=spec[key]
            combos.append(arm)
            if len(combos)==2:break
        if len(combos)==2:break
    save(ART/'combination_protocol.json',{'beneficial_single_factors':[row['variant'] for row in beneficial],
                                        'selected_combinations':combos,'no_more_than_two':True})
    for arm in combos:
        steps=1536 if arm['batch']==256 else 768
        allrows.update(fresh(arm['name'],[arm],steps=steps,checkpoints=[steps*2//3],workers=workers))
    rankedall=sorted((compare(row,extcontrol) for row in allrows.values()),key=lambda row:(not row['eligible'],row['R'],row['variant']))
    shortlist=['new_control']
    for row in rankedall:
        if row['variant'] not in shortlist:
            shortlist.append(row['variant']);break
    attrsorted=sorted(rankedall,key=lambda row:(not row['eligible'],row['R_attr'],row['R'],row['variant']))
    for row in attrsorted:
        if row['variant'] not in shortlist:
            shortlist.append(row['variant']);break
    long_arms=[candidate(name,allrows[name]) for name in shortlist]
    save(ART/'long_forecast_protocol.json',{'shortlist':shortlist,'arms':long_arms,
                                          'primary_optimizer_budget_B512':4096,'independent_seeds_after_frozen_pair':True})
    long_rows={}
    for arm in long_arms:
        steps=8192 if arm['batch']==256 else 4096
        result=fresh('long_seed817_'+arm['name'],[arm],steps=steps,
                     checkpoints=[steps//4,steps//2,steps*3//4],scenes=4096,workers=workers,confirmation=True)
        long_rows.update(result)
    all_long=[]
    for arm in long_arms:
        all_long+=read(RUNS/('long_seed817_'+arm['name'])/'summary.json')
    agg={arm['name']:aggregate([row for row in all_long if row['variant']==arm['name'] and
         row['budget']==(8192 if arm['batch']==256 else 4096)]) for arm in long_arms}
    longrank=sorted((dict(compare(values,agg['new_control']),variant=name) for name,values in agg.items()),
                    key=lambda row:(not row['eligible'],row['R'],row['variant']))
    alternatives=[row for row in longrank if row['variant']!='new_control']
    chosen=alternatives[0]['variant']
    finalarms=[next(arm for arm in long_arms if arm['name']==name) for name in ('new_control',chosen)]
    save(ART/'frozen_confirmation_protocol.json',{'arms':finalarms,'preliminary_rank':longrank,
             'candidate':chosen,'control':'new_control','train_seeds':[817,1817,2817],'generation_seeds':[2817,3817],
             'primary':'EMA final equal-exposure endpoint; R ratio of TRAIN arm means, same gates as search_protocol',
             'full50000_no_reselection':True})
    for seed in (1817,2817):
        for arm in finalarms:
            steps=8192 if arm['batch']==256 else 4096
            fresh('confirm_seed'+str(seed)+'_'+arm['name'],[arm],steps=steps,
                  checkpoints=[steps//2,steps*3//4],scenes=4096,seed=seed,workers=workers,confirmation=True)
    final_rows=[]
    for seed in (817,1817,2817):
        for arm in finalarms:
            label=('long_seed817_' if seed==817 else 'confirm_seed'+str(seed)+'_')+arm['name']
            steps=8192 if arm['batch']==256 else 4096
            final_rows += [row for row in read(RUNS/label/'summary.json') if row['budget']==steps]
    means={arm['name']:aggregate([row for row in final_rows if row['variant']==arm['name']]) for arm in finalarms}
    ranked_final=sorted((dict(compare(mean,means['new_control']),variant=name) for name,mean in means.items()),
                        key=lambda row:(not row['eligible'],row['R'],row['variant']))
    improvements=[row for row in ranked_final if row['eligible'] and row['R']<1.0]
    selected=improvements[0]['variant'] if improvements else 'new_control'
    save(ART/'final_selection.json',{'selected':selected,'all':ranked_final,'rows':final_rows,
                                    'selected_arm':next(arm for arm in finalarms if arm['name']==selected),
                                    'decision':'best eligible improvement' if improvements else 'retain control: no eligible improvement',
                                    'global128epoch_optimum_proven':False})
    for arm in finalarms:
        checkpoint_map=DESIGN/('full50000_'+arm['name']+'.json')
        save(checkpoint_map,{arm['name']:long_rows[arm['name']]['checkpoint']})
        command=base(batch=arm['batch'],scenes=50000,normalizer=Path(arm['normalizer']),workers=workers)+[
            '--steps','1','--checkpoint-steps','','--variants','','--evaluation-checkpoints',str(checkpoint_map),
            '--require-full-set','--output',str(RUNS/('full50000_'+arm['name']))]
        run('full50000_'+arm['name'],command)
    save(ART/'queue_complete.json',{'selected':selected,'status':'COMPLETE_ACTUAL_SEARCH',
                                  'finished_unix':time.time(),'full_set_audits':2})
    save(ART/'queue_active.json',{'stage':'COMPLETE','controller_pid':os.getpid()})
    print('ALL FROZEN MAP SEARCH STAGES DONE',flush=True)


if __name__=='__main__':
    main()
