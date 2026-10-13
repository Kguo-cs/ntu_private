"""Explicit N20 full-run config materialization from verified research evidence.

Default is a dry run: verify native stages and compose temporary Hydra configs.
No winner is selected; no Torch/model is imported or instantiated. Short training
is recipe evidence, never proof of a128epoch final result. Use --write explicitly.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, re, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
import yaml
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import report_init_diffusion_657_compact as report
BASE = 'init_diffusion_657_full_base'
MAP = ROOT/'src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt'
NORMALIZER = ROOT/'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'
PARAMETER_LIMIT = 8_100_000
MODEL_PREFIX = 'model.model_config.'
INIT_NATIVE_FIELDS = (
    'heading_noise','heading_objective','heading_x0_loss','heading_flow_loss_weight',
    'matching_weights','time_sample_power','collision_reduction','fix_ego',
    'fix_ego_position','fix_ego_heading','fix_ego_shape','fix_ego_velocity','fix_ego_type',
    'use_ego_embedding','ego_context_heading_encoding','generate_type','type_loss_weight',
    'type_process','type_match_weight','type_noise_stats','pos_source','shape_source',
    'velocity_source','velocity_representation','speed_loss_weight','speed_loss_scale',
    'size_representation','invalid_size_policy','sampling_steps','edge_embedding_type',
    'time_embedding_type','time_embedding_scale','count_embedding_type','count_lane_source',
    'count_max_num_agents','count_max_num_lanes','map_embedding_type','map_id_source',
    'map_id','map_lg_type','map_label_dropout','use_ema','ema_decay')
PHYSICAL_DATA_FIELDS = ('train_raw_dir','val_raw_dir','test_raw_dir','val_tfrecords_splitted',
    'scenario_dreamer_preprocessed','scenario_dreamer_train_non_partitioned_only',
    'scenario_dreamer_train_latent_cache','scenario_dreamer_val_latent_cache','scenario_dreamer_test_latent_cache')

def require(condition, message):
    if not condition: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def proof(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def plain(value): return OmegaConf.to_container(value,resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)

def portable(value):
    if isinstance(value,str) and value.startswith(str(ROOT)+'/'): return '${sim_root:}/'+str(Path(value).relative_to(ROOT))
    if isinstance(value,dict): return {k:portable(v) for k,v in value.items()}
    if isinstance(value,list): return [portable(v) for v in value]
    return value

def select_evidence(data, identity, *, tier=None, budget=None):
    require(data.get('report_kind')=='combined_verified_main_and_compact','Expected combined verified report')
    require(identity.count(':')==1,'--variant must be source:variant')
    source,variant=identity.split(':'); require(source in ('main','compact') and variant,'Only main/compact native variants accepted')
    candidates=[v for v in data['comparisons'] if v['source']==source and v['variant']==variant
        and (tier is None or v['tier']==tier) and (budget is None or v['budget']==budget)
        and v['evidence'].get('complete_terminal_evidence') is True]
    require(candidates,'No completed terminal evidence for requested source/variant/tier/budget')
    rank=max(report.TIERS.index(v['tier']) for v in candidates)
    candidates=[v for v in candidates if report.TIERS.index(v['tier'])==rank]
    require(len(candidates)==1,'Multiple completed cohorts; specify --tier/--budget, never pool different scopes')
    selected=copy.deepcopy(candidates[0]); rows=selected['native_observations']
    require(rows and json.loads(json.dumps(report.evidence_completeness(rows)))==selected['evidence'],'Declared terminal/seed grid differs from native observations')
    counts={r.get('generator_num_parameters') for r in rows}
    require(len(counts)==1 and all(type(n) is int and 0<n<=PARAMETER_LIMIT for n in counts),'Missing/inconsistent actual G1 count or exceeds8.1M')
    require(selected['resources']['actual_generator_parameters']==sorted(counts),'Declared G1 count differs from native')
    require(selected['primary_fresh657'].get('available') is True,'No exact-scope verified fresh657 denominator')
    return selected

def verify_report_snapshot(data, selected):
    sources=data['sources']
    rebuilt=json.loads(json.dumps(report.build_data(main_art=Path(sources['main']['artifact_dir']),main_runs=Path(sources['main']['run_dir']),
        compact_art=Path(sources['compact']['artifact_dir']),compact_runs=Path(sources['compact']['run_dir']))))
    for name in ('main','compact'):
        require(sources[name].get('protocol_sha256')==rebuilt['sources'][name].get('protocol_sha256'),'Report source protocol changed: '+name)
    keys=('source','variant','tier','budget','num_samples','eval_manifest_sha256')
    matches=[v for v in rebuilt['comparisons'] if all(v[k]==selected[k] for k in keys)]
    require(len(matches)==1,'Cohort no longer hash-verified; refresh combined report')
    for key in ('evidence','native_observations','primary_fresh657','per_TRAIN','resources'):
        require(matches[0][key]==selected[key],'Stale/changed report evidence: '+key)
    return rebuilt['sources'][selected['source']]

def registered_variant(source, name):
    art=Path(source['artifact_dir']); path=art/'variants.json'; variants=read(path)
    require(name in variants and isinstance(variants[name],dict),'Variant absent from native registry')
    variant=variants[name]; plan=source['protocol']['effective_plan']; registration_proof=[]
    if source['source']=='compact':
        path_plan=art/'effective_plan.json'; plan=read(path_plan); registration_proof.append(proof(path_plan))
        require(plan['variants'].get(name)==variant,'Compact effective plan and variant registry disagree')
    elif name not in plan['variants']:
        path_plan=art/'adaptive_registration.json'; adaptive=read(path_plan); registration_proof.append(proof(path_plan))
        require(adaptive.get('name')==name and adaptive.get('variant')==variant and adaptive.get('status')=='REGISTERED',
            'Adaptive graph variant lacks exact registered provenance')
        require(adaptive['sources']==source['protocol']['sources'] and adaptive['plan_sha256']==source['protocol']['plan_sha256'],
            'Adaptive registration source/plan mismatch')
    else: require(plan['variants'][name]==variant,'Variant changed from frozen plan')
    require(all(k.startswith(MODEL_PREFIX) for k in variant.get('__config',{})),'Unsupported non-model variant delta')
    return copy.deepcopy(variant),{'registry':proof(path),'extra_registration':registration_proof}

def apply_variant(recipe, variant):
    result=copy.deepcopy(recipe)
    for name,value in variant.items():
        if name.startswith('__'): require(name in ('__config','__lr','__clip'),'Unknown variant field: '+name)
        elif name in ('heading_space','std_floor','normalize_time_weights'): result[name]=value
        elif name=='components':
            for component,changes in value.items(): result['components'][component].update(changes)
        else: result['components'][name].update(value)
    return result

def recipe_matches(actual, expected):
    expected=copy.deepcopy(expected)
    expected['components'].setdefault('speed_magnitude',dict(kind='mse',time_power=0,weight=0.,scale='data'))
    return actual==expected

def validate_native_config(cfg, native, row, variant, registered_recipe):
    model=cfg['model']['model_config']; decoder=model['decoder']; init=decoder['init_diffusion']; actual=native['decoder']
    require(cfg['model']['_target_']=='src.smart.model.smart_gail.SMART_GAIL','Unexpected model target')
    require(cfg['action']=='fit' and cfg['ckpt_path'] is None and cfg['seed']==row['seed'],'Not fresh fitting with recorded TRAIN seed')
    require(model['initial_map_checkpoint']==str(MAP) and model['finetune'] is True and row['map_frozen'] is True,'Not the exact frozen AIRL map')
    require(row['initialization_scope']=='frozen_pretrained_map_fresh_G1' and row['batch_size']==512,'Not batch512 freshG1')
    require(row['budget']==actual['ema_num_updates'] and row['evaluation_weights']=='ema','Native checkpoint/EMA exposure differs')
    require(cfg['trainer']['precision']==native['runtime']['trainer_precision']=='32-true'
        and cfg['float32_matmul_precision']==native['runtime']['float32_matmul_precision']=='medium','Precision differs')
    validate_full_data(cfg)
    require(cfg['data']['train_batch_size']==512 and cfg['trainer']['accumulate_grad_batches']==1 and cfg['data']['shuffle'] is False,
        'Quick-source batch/accumulation/repeated-order scope differs')
    require(decoder['init_decoder']=='flow' and decoder['sep_map'] is False and decoder['initial_scene_only'] is True,'Not shared-map supervised InitDiffusion')
    require(all(model['token_processor'].get(k) is True for k in ('pred_init','learn_init','scenario_dreamer_init'))
        and model['token_processor']['learn_autoencoder'] is False,'Source token supervision differs')
    require(init.get('normalizer_path') is None,'Quick source unexpectedly used startup normalizer')
    require(native['map_source']=='reference' and native['reference_mode']=='matched' and native['unverified_generation_timestep_count']==0,'Map/reference/membership differs')
    require(actual['name']=='InitDiffusion' and actual['mode']=='lane_conditioned' and actual['sampling_steps']==20,'Initializer/sampler differs')
    require(recipe_matches(actual['component_loss'],apply_variant(registered_recipe,variant))
        and recipe_matches(actual['component_loss'],init['component_loss']),'Native loss differs from registered/source recipe')
    for path,value in variant.get('__config',{}).items():
        require(OmegaConf.select(OmegaConf.create(cfg),path)==value,'Source config differs from variant: '+path)
    require(model['lr']==row['lr']==variant.get('__lr',.0005),'Actual LR differs')
    require(cfg['trainer']['gradient_clip_val']==row['gradient_clip_val']==variant.get('__clip',.5),'Actual clip differs')
    require(model['optimizer']=='default' and model['lr_warmup_steps']==0 and model['lr_total_steps']==128 and model['lr_min_ratio']==.05,'Optimizer/epoch scheduler differs')
    for key,value in actual['denoiser_options'].items():
        require(init.get(key,'input' if key=='time_conditioning' else None)==value,'Source denoiser setting differs: '+key)
    for key in INIT_NATIVE_FIELDS:
        if key in actual and key in init: require(init[key]==actual[key],'Source/native initializer flag differs: '+key)
    require(model['token_processor']['init_map_crop']==actual['init_map_crop']=='square'
        and model['token_processor']['init_map_half_extent']==actual['init_map_half_extent_m']==50.,'Physical crop differs')

def validate_full_data(cfg):
    data=cfg['data']; model=cfg['model']['model_config']
    expected={'train_raw_dir':str(ROOT/'src/waymo_data/full/training_map2_sd'),
              'val_raw_dir':str(ROOT/'src/waymo_data/full/scenario_dreamer_val'),
              'test_raw_dir':str(ROOT/'src/waymo_data/full/scenario_dreamer_val')}
    require(all(data.get(k)==v for k,v in expected.items()),'Physical full TRAIN/val/test cache directories differ')
    require(data.get('val_tfrecords_splitted') is None and data.get('scenario_dreamer_preprocessed') is False
        and data.get('scenario_dreamer_train_non_partitioned_only') is False
        and all(data.get(k) is None for k in ('scenario_dreamer_train_latent_cache','scenario_dreamer_val_latent_cache','scenario_dreamer_test_latent_cache')),
        'Full cache path retained a partition/raw/latent-data restriction')
    require(model.get('sd_use_cached_evaluator') is True and model.get('sd_gen_timestep')==0
        and model.get('sd_prediction_frame')=='world' and model.get('sd_require_generation_timestep') is True,
        'Official initial-scene evaluator/membership flags differ')

def validate_calibration(calibration):
    require(calibration['scenes']==8192 and calibration['source'].startswith('training_map2_sd'),'Expected TRAIN-only8192 calibration')
    for key in ('normal_mean','normal_scale'):
        values=calibration[key]; require(isinstance(values,list) and len(values)==1 and len(values[0])==7,'Calibration shape must be1x7')
        require(all(type(v) in (int,float) and math.isfinite(v) and (key!='normal_scale' or v>0) for v in values[0]),'Invalid calibration values')

def load_native_source(selected, source, variant):
    recipe_path=Path(source['artifact_dir'])/'recipe.json'; recipe=read(recipe_path)
    configs=[]; natives=[]; provenance=[]
    for row in selected['native_observations']:
        native=read(row['metrics_path']); require(sha(row['metrics_path'])==row['metrics_sha256'],'Native metrics changed')
        path=Path(row['checkpoint']).parent/'config.yaml'; cfg=plain(OmegaConf.load(path))
        validate_native_config(cfg,native,row,variant,recipe); configs.append(cfg); natives.append(native)
        provenance.append({'TRAIN':row['seed'],'GEN':row['eval_seed'],'source_config':proof(path),'metrics':proof(row['metrics_path']),'checkpoint':row['checkpoint']})
    for cfg in configs[1:]: require(cfg['model']==configs[0]['model'],'Native model differs between TRAIN/GEN cells')
    for native in natives[1:]: require(native['decoder']==natives[0]['decoder'],'Native decoder differs between cells')
    validate_calibration(read(NORMALIZER)); norm_sha=sha(NORMALIZER); map_sha=sha(MAP)
    require(all(r['calibration_sha256']==norm_sha and r['map_checkpoint_sha256']==map_sha for r in selected['native_observations']),'Calibration/map file changed')
    return configs[0],natives[0],{'configs_and_metrics':provenance,'registered_recipe':proof(recipe_path),
        'calibration':{'path':str(NORMALIZER),'sha256':norm_sha},'fixed_map':{'path':str(MAP),'sha256':map_sha}}

def build_configs(cfg, native, variant, name, *, train_seed=4817, gen_seed=2817, eval_checkpoint=None):
    require(re.fullmatch(r'[a-z][a-z0-9_]*',name) is not None,'Output name requires lowercase letters/digits/underscores')
    model=portable(copy.deepcopy(cfg['model'])); settings=model['model_config']; init=settings['decoder']['init_diffusion']
    init.update(copy.deepcopy(native['decoder']['denoiser_options'])); init['component_loss']=copy.deepcopy(native['decoder']['component_loss'])
    for key in INIT_NATIVE_FIELDS:
        if key in native['decoder']: init[key]=copy.deepcopy(native['decoder'][key])
    init['normalizer_path']=portable(str(NORMALIZER))
    settings.update(initial_map_checkpoint=portable(str(MAP)),finetune=True,lr_total_steps=128,sd_r_reference=None,
        sd_reference_mode='matched',sd_require_full_set=False,sd_expected_scenes=50000)
    train={'defaults':[BASE,'_self_'],'task_name':name,'action':'fit','ckpt_path':None,
        'seed':train_seed,'float32_matmul_precision':'medium','model':model,
        'data':{'train_batch_size':512,'val_batch_size':128,'test_batch_size':128,
            **{k:portable(cfg['data'][k]) for k in PHYSICAL_DATA_FIELDS if k in cfg['data']},
            'shuffle':True,'scenario_dreamer_train_sample_list':None,'scenario_dreamer_eval_set':None,
            'num_workers':0,'pin_memory':False,'persistent_workers':False},
        'trainer':{'max_epochs':128,'max_steps':-1,'precision':'32-true','accumulate_grad_batches':1,
            'gradient_clip_val':float(cfg['trainer']['gradient_clip_val'])}}
    evaluate={'defaults':[name,'_self_'],'task_name':name+'_eval','action':'test',
        'ckpt_path':portable(str(Path(eval_checkpoint).resolve())) if eval_checkpoint else '${oc.env:'+name.upper()+'_CKPT}',
        'seed':gen_seed,'model':{'model_config':{'initial_map_checkpoint':None,'sd_reference_mode':'matched',
            'sd_require_full_set':True,'sd_expected_scenes':50000,'sd_r_reference':None,
            'decoder':{'init_diffusion':{'normalizer_path':None}}}},
        'data':{'scenario_dreamer_eval_set':None},'trainer':{'limit_test_batches':1.,'logger':False,'enable_checkpointing':False},
        'callbacks':None,'logger':False}
    return train,evaluate

def config_text(cfg, *, evaluation=False):
    return ('# @package _global_\n# Verified short-training recipe evidence; no128epoch final-result claim.\n'
        + ('# Supply a new full-training checkpoint; restores map/G1/calibration/EMA.\n' if evaluation else
           '# FreshG1/optimizer/EMA; fixed AIRL map; TRAIN8192 calibration; epoch shuffle enabled.\n')
        + '# Use optimizer=default in the run command; optimizer groups are applied after experiments.\n'
        + '# Seven native metrics; final R needs matched128epoch/50k/TRAIN-GEN baseline supplied explicitly.\n'
        + yaml.safe_dump(cfg,sort_keys=False))

def compose_configs(train, evaluate, name):
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
    with tempfile.TemporaryDirectory(prefix='init657-n20-compose-') as temporary:
        cfgdir=Path(temporary)/'configs'; shutil.copytree(ROOT/'configs',cfgdir)
        for item_name,value in ((name,train),(name+'_eval',evaluate)):
            (cfgdir/'experiment'/(item_name+'.yaml')).write_text(config_text(value,evaluation=item_name.endswith('_eval')))
        with initialize_config_dir(config_dir=str(cfgdir),version_base=None):
            overrides=['optimizer=default','paths.output_dir='+temporary,'paths.log_dir='+temporary]
            fit=plain(compose(config_name='run',overrides=['experiment='+name,*overrides]))
            test=plain(compose(config_name='run',overrides=['experiment='+name+'_eval',*overrides,'ckpt_path=/explicit/new_full_training_checkpoint.ckpt']))
    return fit,test

def validate_composed(fit, test, train, native, variant):
    validate_full_data(fit); validate_full_data(test)
    model=fit['model']['model_config']; init=model['decoder']['init_diffusion']
    require(fit['model']==plain(OmegaConf.create(train['model'])),'Hydra changed selected full model')
    require(fit['action']=='fit' and fit['ckpt_path'] is None and model['initial_map_checkpoint']==str(MAP)
        and model['finetune'] is True and init['normalizer_path']==str(NORMALIZER),'Fit lost map-only fresh initialization/calibration')
    require(fit['trainer']['max_epochs']==128 and fit['trainer']['max_steps']==-1 and model['lr_total_steps']==128,'Not128epoch fit/scheduler')
    require(fit['data']['shuffle'] is True and fit['data']['train_batch_size']==512
        and fit['data']['scenario_dreamer_train_sample_list'] is None and fit['data']['scenario_dreamer_eval_set'] is None,'Full training kept quick subset/order')
    require(fit['trainer']['precision']=='32-true' and fit['trainer']['accumulate_grad_batches']==1 and fit['float32_matmul_precision']=='medium','Composed precision differs')
    require(model['optimizer']=='default' and model['lr']==variant.get('__lr',.0005) and model['lr_warmup_steps']==0
        and model['lr_min_ratio']==.05 and fit['trainer']['gradient_clip_val']==variant.get('__clip',.5),'Final optimizer group changed recipe')
    require(init['component_loss']==native['decoder']['component_loss'] and init['sampling_steps']==20 and init['use_ema'] is True and init['ema_decay']==.9999,'Composed loss/EMA/N20 differs')
    for key,value in native['decoder']['denoiser_options'].items(): require(init[key]==value,'Composed denoiser differs: '+key)
    for path,value in variant.get('__config',{}).items(): require(OmegaConf.select(OmegaConf.create(fit),path)==value,'Composed delta differs: '+path)
    require(test['action']=='test' and test['ckpt_path']=='/explicit/new_full_training_checkpoint.ckpt'
        and test['model']['model_config']['initial_map_checkpoint'] is None
        and test['model']['model_config']['decoder']['init_diffusion']['normalizer_path'] is None,'Eval lost explicit checkpoint or overwrites saved map/calibration')
    require(test['model']['model_config']['sd_require_full_set'] is True and test['model']['model_config']['sd_expected_scenes']==50000
        and test['trainer']['limit_test_batches']==1. and test['data']['scenario_dreamer_eval_set'] is None,'Eval not complete50k membership check')
    left=copy.deepcopy(fit['model']); right=copy.deepcopy(test['model'])
    for item in (left,right):
        for key in ('initial_map_checkpoint','sd_require_full_set'): item['model_config'].pop(key,None)
        item['model_config']['decoder']['init_diffusion'].pop('normalizer_path',None)
    require(left==right,'Train/eval model or conditioning differs')
    require(model['sd_r_reference'] is None and test['model']['model_config']['sd_r_reference'] is None,'Unmatched historical R denominator leaked')

def write_outputs(outputs, *, overwrite=False):
    for path,text in outputs.items(): require(overwrite or not path.exists() or path.read_text()==text,'Refusing overwrite: '+str(path))
    for path,text in outputs.items(): report.atomic_text(path,text)

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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=report.ART/'live_report.json',help='Combined verified live_report.json')
    parser.add_argument('--variant',required=True,help='Explicit main:variant or compact:variant; no winner selection')
    parser.add_argument('--output-name',required=True); parser.add_argument('--tier',choices=report.TIERS); parser.add_argument('--budget',type=int)
    parser.add_argument('--train-seed',type=int,default=4817); parser.add_argument('--gen-seed',type=int,default=2817)
    parser.add_argument('--eval-checkpoint',type=Path,help='New full-training checkpoint; quick evidence checkpoints prohibited')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'configs/experiment')
    parser.add_argument('--evidence-dir',type=Path,default=ROOT/'output/reports/init_diffusion_657_n20_configs')
    mode=parser.add_mutually_exclusive_group(); mode.add_argument('--write',action='store_true'); mode.add_argument('--dry-run',action='store_true',help='Default: verify/compose but write nothing')
    parser.add_argument('--overwrite',action='store_true'); args=parser.parse_args(argv)
    source_bytes=args.source.read_bytes(); data=json.loads(source_bytes); source_proof={'path':str(args.source.resolve()),'sha256':hashlib.sha256(source_bytes).hexdigest()}; selected=select_evidence(data,args.variant,tier=args.tier,budget=args.budget)
    source=verify_report_snapshot(data,selected); variant,variant_proof=registered_variant(source,selected['variant'])
    cfg,native,native_proof=load_native_source(selected,source,variant)
    if args.eval_checkpoint:
        require(str(args.eval_checkpoint.resolve()) not in {r['checkpoint'] for c in data['comparisons'] for r in c['native_observations']},'Quick checkpoint is evidence, not128epoch deployment')
    train,evaluate=build_configs(cfg,native,variant,args.output_name,train_seed=args.train_seed,gen_seed=args.gen_seed,eval_checkpoint=args.eval_checkpoint)
    fit,test=compose_configs(train,evaluate,args.output_name); validate_composed(fit,test,train,native,variant)
    evidence={'status':'HYDRA_AND_NATIVE_VERIFIED','created_utc':datetime.now(timezone.utc).isoformat(),
        'identity':args.variant,'source_report':source_proof,'variant_registry':variant_proof,
        'source_protocol_sha256':source['protocol_sha256'],'source_bundle_sha256':source['source_bundle_sha256'],
        'native_provenance':native_proof,'evidence_cohort':selected,'registered_variant':variant,
        'native_component_loss':native['decoder']['component_loss'],'actual_G1_parameters':selected['resources']['actual_generator_parameters'][0],
        'parameter_limit':PARAMETER_LIMIT,'full_base':proof(ROOT/'configs/experiment'/(BASE+'.yaml')),'helper':proof(Path(__file__)),
        'model_instantiated':False,'torch_imported':'torch' in sys.modules,
        'full_training':{'epochs':128,'max_steps':-1,'TRAIN':args.train_seed,'batch_size':512,'sampling_steps':20,'EMA':.9999,
            'precision':'32-true','matmul':'medium','shuffle':True,'initialization':'FreshG1/optimizer/EMA; only fixed AIRL map restored; TRAIN8192 calibration'},
        'full_evaluation':{'scenes':50000,'require_full_set':True,'GEN':args.gen_seed,'weights':'EMA','search_checkpoint_used':False,'checkpoint_policy':'Requires a new full-training checkpoint; known quick-search checkpoint paths are rejected.',
            'checkpoint_content_verified':False,'required_checkpoint_training_epochs':128,'R_reference':None},
        'short_training_not_128epoch_final':True,'global_best_claim':False,
        'differences':['Full training reshuffles each epoch; quick evidence repeated one seeded order.',
            'Recorded R belongs only to its source/TRAIN/GEN/exposure/manifest/fresh-control denominator.',
            'No cross-cohort unified best is claimed; only explicitly requested recipe is materialized.',
            'Final R requires a matched128epoch/50k/TRAIN-GEN native baseline report via explicit model.model_config.sd_r_reference override.',
            'Eval checkpoint content is not inspected by this CPU config helper; confirm its complete128epoch training provenance before evaluation.'],
        'commands':{'train':'python src/run.py experiment='+args.output_name+' optimizer=default',
            'evaluate':'python src/run.py experiment='+args.output_name+'_eval optimizer=default ckpt_path=/path/to/new_full_training.ckpt'}}
    require(not evidence['torch_imported'],'CPU helper unexpectedly imported Torch/model code')
    if args.write:
        snapshot=freeze_input_snapshot(source_bytes,args.evidence_dir/'materializer_inputs')
        require(snapshot['sha256']==source_proof['sha256'],'Captured input SHA differs from verified source')
        evidence['source_report']={'path':snapshot['path'],'sha256':snapshot['sha256'],'origin_path':source_proof['path'],'snapshot':snapshot}
    paths={args.output_dir/(args.output_name+'.yaml'):config_text(train),args.output_dir/(args.output_name+'_eval.yaml'):config_text(evaluate,evaluation=True),
        args.evidence_dir/(args.output_name+'.evidence.json'):json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n'}
    if args.write: write_outputs(paths,overwrite=args.overwrite)
    print(json.dumps({'status':'WRITTEN' if args.write else 'DRY_RUN_VERIFIED_NO_OUTPUT','identity':args.variant,
        'tier':selected['tier'],'budget':selected['budget'],'TRAIN':selected['TRAIN'],'GEN':selected['GEN'],
        'actual_G1_parameters':evidence['actual_G1_parameters'],'R':selected['primary_fresh657']['R'],'gates':selected['primary_fresh657']['gates'],
        'short_training_not_128epoch_final':True,'torch_imported':False,'paths':[str(p) for p in paths]},ensure_ascii=False,indent=2))
    return evidence

if __name__=='__main__': main()
