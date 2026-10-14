"""Separate new-method effects from their shared matching4 parent recipe."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_expanded_20261014 as search


def main():
    rows=search.read(search.ART/'observations.json');result=[]
    cells=sorted({(r['variant'],r['seed'],r['budget'],r['num_samples']) for r in rows})
    for arm,seed,budget,scenes in cells:
        if arm in (search.BASE,'match4'):continue
        candidate=[r for r in rows if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,seed,budget,scenes)]
        parents=['match4']+(['prior_speed'] if arm=='prior_shape_speed' else [])
        if arm=='match4_affine_heading_p1':parents+=['match4_shape_affine','match4_heading_p1']
        for parent in parents:
            control=[r for r in rows if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(parent,seed,budget,scenes)]
            if len(candidate)!=2 or len(control)!=2:continue
            score=search.paired_score(candidate,control)
            result.append(dict(arm=arm,parent=parent,seed=seed,budget=budget,scenes=scenes,**score))
    search.save(search.ART/'parent_comparisons.json',dict(generated_utc=search.now(),comparisons=result,
        scope='Same budget/TRAIN/GEN/cohort.64-step pilots excluded from efficacy claims.',
        source_observations_sha256=search.sha(search.ART/'observations.json')))
    print([(r['arm'],r['parent'],r['budget'],round(r['R_focus'],4)) for r in result if r['budget']>=2048])


if __name__=='__main__':main()
