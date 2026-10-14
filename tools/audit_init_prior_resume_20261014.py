"""Trace affected old quick-run checkpoints; preserve their original evidence."""
from __future__ import annotations
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'
OLD_ROOTS=[ROOT/'logs'/name for name in ('init_diffusion_evening_20261013','init_diffusion_overnight_20261014')]


def main():
    stages=[]
    for path in sorted(p for root in OLD_ROOTS for p in root.rglob('summary.json')):
        rows=read(path)
        if not isinstance(rows,list) or not rows:continue
        row=rows[0];folder=Path(row['checkpoint']).parent
        manifest=folder.parent/'train_order.txt'
        with manifest.open() as stream:scenes=sum(1 for _ in stream)
        batches=math.ceil(scenes/row['batch_size'])
        trace_path=folder/'train_trace.json'
        trace=read(trace_path) if trace_path.exists() and not row['evaluation_only'] else []
        first_error=None
        for cell in trace:
            expected=row['lr']*(.05+.95*.5*(1+math.cos(math.pi*min(max(1,cell['step']//batches)/128,1.))))
            if not math.isclose(cell['learning_rate'],expected,rel_tol=1e-12,abs_tol=1e-15):
                first_error=dict(step=cell['step'],actual=cell['learning_rate'],expected=expected);break
        resumed=bool(row.get('resume_cyclic_epochs') and not row['evaluation_only'])
        start=row.get('resume_global_step',0)
        first_boundary=(start//batches+1)*batches
        repeated_suffix=resumed and start%batches!=0 and row['budget']>first_boundary
        stages.append(dict(summary=str(path),summary_sha256=sha(path),arm=row['variant'],
            TRAIN=row['seed'],updates=row['budget'],scenes=row['num_samples'],cells=len(rows),
            checkpoint=row['checkpoint'],source_checkpoint=row.get('resumed_from'),
            evaluation_only=row['evaluation_only'],cyclic_resume=resumed,resume_step=start,
            training_scenes=scenes,batches_per_epoch=batches,
            first_lr_mismatch=first_error,repeated_suffix_by_original_loader=repeated_suffix,
            invalid_reasons=(['EPOCH_SUFFIX_REPEATED'] if repeated_suffix else [])+
                           (['WRONG_LR_FOR_FULL_DATA_EPOCH'] if first_error and resumed else [])))
    bad={s['checkpoint'] for s in stages if s['invalid_reasons'] and not s['evaluation_only']}
    while True:
        previous=len(bad)
        for stage in stages:
            if stage['source_checkpoint'] in bad:
                if 'SOURCE_CHECKPOINT_VIOLATES_TRAINING_PROTOCOL' not in stage['invalid_reasons']:
                    stage['invalid_reasons'].append('SOURCE_CHECKPOINT_VIOLATES_TRAINING_PROTOCOL')
                if not stage['evaluation_only']:bad.add(stage['checkpoint'])
        if len(bad)==previous:break
    affected=[s for s in stages if s['invalid_reasons']]
    result=dict(at_utc=now(),status='PRIOR_EVIDENCE_CORRECTION',script_sha256=sha(__file__),
        source_introduced_commit='9fe23e6af504872836c8da38fb1a9567a48f9e0e',
        root_cause='Singular reload_dataloaders_every_n_epoch is unused; resumed quick-runner repeats its first dataset suffix. Complete default src/run.py training does not call this helper.',
        affected_stages=len(affected),affected_cells=sum(s['cells'] for s in affected),
        affected_full50000_stages=sum(s['evaluation_only'] and s['scenes']==50000 for s in affected),
        scope='Old evening/overnight quick-search resumed results and evaluations of their descendant checkpoints do not satisfy the registered original full-epoch training protocol. Their numerical generation metrics remain raw measurements of those checkpoints, but cannot support the original training-exposure or full-training prediction claims. Fresh runs and independent W&B full training are not invalidated by this helper bug.',
        action='Preserve original files with this correction. Exclude affected stages from current selection. Current expanded search restarts valid2048 trajectories under exact-resume regression and native loader/LR auditing.',
        stages=stages)
    save(ART/'prior_resume_evidence_correction.json',result)
    for parent in ('init_diffusion_overnight_20261014','init_diffusion_overnight_20261014/recovery',
                   'init_diffusion_overnight_20261014/adaptive_long','init_diffusion_overnight_20261014/final_verification',
                   'init_diffusion_evening_20261013','init_diffusion_evening_20261013/recovery',
                   'init_diffusion_evening_20261013/long_audit/recovery','init_diffusion_evening_20261013/final_long',
                   'init_diffusion_evening_20261013/final_long/rebased','init_diffusion_evening_20261013/final_long/rebased/sampler_merge'):
        folder=ROOT/'output/reports'/parent
        if folder.exists():save(folder/'continuation_audit_notice.json',dict(at_utc=now(),
            status=result['status'],evidence=str(ART/'prior_resume_evidence_correction.json'),
            evidence_sha256=sha(ART/'prior_resume_evidence_correction.json'),scope=result['scope']))
    print({k:result[k] for k in ('status','affected_stages','affected_cells','affected_full50000_stages')})


if __name__=='__main__':main()
