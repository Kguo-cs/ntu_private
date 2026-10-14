"""Independently audit completed architecture cells, schedules and live options."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import audit_init_overnight_20261014 as metric
from tools import audit_init_expanded_schedule_20261014 as schedule
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014/map_architecture'

def main():
    path=ART/'observations.json'
    if not path.exists() or not read(path):
        print('NO_COMPLETED_ARCHITECTURE_CELLS');return
    rows=read(path)
    audit=metric.audit_rows(rows)
    audit.update(observations_sha256=sha(path),phase='MAP_DENOISER_ARCHITECTURE')
    save(ART/'native_metric_audit.json',audit)
    stages={}
    effective=[]
    for row in rows:
        if not row['evaluation_only']:
            key=(row['variant'],row['seed'],row['budget'])
            stages.setdefault(key,row)
        report=read(row['metrics_path'])
        options=report['decoder'].get('denoiser_options')
        # Configuration is retained even if an older metadata collector omits
        # optional dimensions; preflight proved these fields reach real modules.
        config=Path(row['checkpoint']).parent/'config.yaml'
        import yaml
        cfg=yaml.safe_load(config.read_text())
        decoder=cfg['model']['model_config']['decoder']
        effective.append(dict(arm=row['variant'],TRAIN=row['seed'],GEN=row['eval_seed'],
            updates=row['budget'],scenes=row['num_samples'],generator_parameters=row['generator_num_parameters'],
            declared_map={k:decoder[k] for k in ('hidden_dim','num_freq_bands','num_heads','head_dim','num_map_layers','pl2pl_radius','pt2pt_neighbor')},
            declared_denoiser={k:decoder['init_diffusion'].get(k) for k in
                ('denoiser_hidden_dim','denoiser_num_layers','denoiser_num_freq_bands','denoiser_num_heads','denoiser_head_dim','map_condition_types')},
            reported_denoiser_options=options,config_sha256=sha(config)))
    schedules=[schedule.audit_row(row) for row in stages.values()]
    save(ART/'training_schedule_audit.json',dict(status='PASS',at_utc=now(),
        observations_sha256=sha(path),stages=len(schedules),audits=schedules,
        checked_updates=sum(r['checked_updates'] for r in schedules)))
    save(ART/'effective_options.json',dict(status='PASS',at_utc=now(),observations_sha256=sha(path),
        preflight_sha256=sha(ART/'prepared_preflight.json'),cells=effective,
        caveat='Declared config, observed parameter counts and optional runtime metadata. CPU actual-module option/gradient checks are recorded separately; missing runtime metadata is not fabricated.'))
    print({'status':audit['status'],'cells':audit['unique_actual_cells'],'checks':audit['checks'],'training_stages':len(schedules)})

if __name__=='__main__':main()
