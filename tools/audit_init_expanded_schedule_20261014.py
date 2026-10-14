"""Verify actual quick-training LR traces against the unchanged full schedule."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'


def audit_row(row):
    key=(row['variant'],row['seed'],row['budget'])
    folder=Path(row['checkpoint']).parent
    cfg=yaml.safe_load((folder/'config.yaml').read_text())
    model=cfg['model']['model_config']
    assert model['optimizer']=='default'
    assert model['lr_total_steps']==128 and model['lr_warmup_steps']==0 and model['lr_min_ratio']==.05
    manifest=folder.parent/'train_order.txt'
    with manifest.open() as stream:scenes=sum(1 for _ in stream)
    batches=math.ceil(scenes/row['batch_size'])
    trace=read(folder/'train_trace.json')
    assert trace[-1]['step']==row['budget']-1
    errors=[]
    for cell in trace:
        # Default profile is epoch-based despite the lr_total_steps name.
        # Initialization and completion of epoch0 both apply epoch index1.
        index=max(1,cell['step']//batches)
        expected=row['lr']*(.05+.95*.5*(1+math.cos(math.pi*min(index/128,1.))))
        error=abs(cell['learning_rate']-expected);errors.append(error)
        assert math.isclose(cell['learning_rate'],expected,rel_tol=1e-12,abs_tol=1e-15),(key,cell['step'],cell['learning_rate'],expected)
    loader_audit = None
    if row['optimizer_continued']:
        events = read(folder/'resume_loader_events.json')
        first_epoch, first_batch = divmod(row['resume_global_step'], batches)
        last_epoch = (row['budget'] - 1) // batches
        assert [e['epoch'] for e in events] == list(range(first_epoch,last_epoch+1)), events
        for event in events:
            first = event['epoch'] == first_epoch
            offset = first_batch * row['batch_size'] if first else 0
            assert event['offset'] == offset
            assert event['dataset_scenes'] == scenes - offset
            assert event['batches'] == math.ceil((scenes-offset)/row['batch_size'])
            assert event['global_step'] == (row['resume_global_step'] if first else event['epoch']*batches)
        loader_audit = dict(events=events,sha256=sha(folder/'resume_loader_events.json'))
    return dict(loader_audit=loader_audit,arm=row['variant'],TRAIN=row['seed'],updates=row['budget'],
        training_scenes=scenes,batches_per_data_pass=batches,data_passes=row['budget']/batches,
        first_trace_step=trace[0]['step'],checked_updates=len(trace),last_learning_rate=trace[-1]['learning_rate'],
        maximum_absolute_lr_error=max(errors),trace_sha256=sha(folder/'train_trace.json'),
        config_sha256=sha(folder/'config.yaml'),manifest_sha256=sha(manifest))


def main():
    rows=read(ART/'observations.json');seen=set();results=[]
    for row in rows:
        key=(row['variant'],row['seed'],row['budget'])
        if key in seen or row['budget']<2048 or row['evaluation_only']:continue
        seen.add(key)
        results.append(audit_row(row))
    save(ART/'training_schedule_audit.json',dict(status='PASS',at_utc=now(),script_sha256=sha(__file__),
        observations_sha256=sha(ART/'observations.json'),results=results,
        interpretation='Verifies actual completed native trace LR values, including resumed stages, '
        'against128epoch default-profile schedule. Native runner repeats the fixed seeded training order; '
        'full recipe reshuffles each epoch. Same schedule does not establish final128epoch metric prediction.'))
    print({'status':'PASS','stages':len(results),'checked_updates':sum(r['checked_updates'] for r in results)})


if __name__=='__main__':main()
