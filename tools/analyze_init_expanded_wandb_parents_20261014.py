"""Compare remote runs with their relevant parents at shared epochs."""
from __future__ import annotations
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now

ART=ROOT/'output/reports/init_diffusion_expanded_20261014'
METRICS=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd')
FOCUS=('speed_jsd','ang_dev_jsd','length_jsd')
BASES=('iu3fa4gx','0vftb3cc','nnsjk5vl')
CANDIDATES=('0vftb3cc','nnsjk5vl','doke6bbr','ehnykqc4','efzhwmu3','wpya0vrc','cchgfuhr','4e89o3nz','4bo59qfn')


def main():
    source=ART/'wandb/latest_analysis.json';analysis=read(source)
    manifest={r['id']:r for r in analysis['manifest']}
    epochs={}
    for rid in set(BASES+CANDIDATES)&manifest.keys():
        groups={}
        for row in read(manifest[rid]['history_file'])['rows']:
            if row.get('epoch') is not None and all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in METRICS):
                groups.setdefault(int(row['epoch']),[]).append(row)
        epochs[rid]={e:{k:statistics.fmean(r[k] for r in rows) for k in METRICS} for e,rows in groups.items()}
    records=[]
    for rid in CANDIDATES:
        if rid not in epochs:continue
        for base in BASES:
            if base==rid or base not in epochs:continue
            common=sorted(epochs[rid].keys()&epochs[base].keys())
            if not common:continue
            for e in common:
                ratios={k:epochs[rid][e][k]/epochs[base][e][k] for k in METRICS}
                records.append(dict(candidate=rid,parent=base,epoch=e,last_shared_epoch=e==common[-1],
                    ratios=ratios,R_focus=statistics.fmean(ratios[k] for k in FOCUS),
                    R_jsd=statistics.fmean(ratios[k] for k in METRICS),
                    R_attr=statistics.fmean(ratios[k] for k in ('speed_jsd','length_jsd','width_jsd')),
                    candidate_validation_scope=manifest[rid]['conditions']['validation_scope'],
                    parent_validation_scope=manifest[base]['conditions']['validation_scope']))
    save(ART/'wandb/current_parent_comparisons.json',dict(generated_utc=now(),
        inventory_fetched_at_utc=analysis['inventory_fetched_at_utc'],source_analysis_sha256=sha(source),analyzer_sha256=sha(__file__),
        comparisons=records,current_constraints=dict(max_sampling_steps=24,parameter_cap=None),
        caveat='Observational remote runs, not local paired ablations. Shared epoch does not establish matching cohorts, GEN seeds, hardware, or source code. Numeric1 validation limits with no membership enforcement remain unverified, not proven to be different scene counts. Larger networks are allowed in the current search; N40 remains outside its sampling budget.'))
    print([dict(candidate=r['candidate'],parent=r['parent'],epoch=r['epoch'],R_focus=round(r['R_focus'],4))
           for r in records if r['last_shared_epoch'] and r['parent']=='nnsjk5vl' and r['candidate'] in ('efzhwmu3','wpya0vrc','cchgfuhr')])


if __name__=='__main__':main()
