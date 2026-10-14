"""Fixed6144-update attribute diagnosis; no new selection metric or causal claim."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.audit_init_overnight_20261014 import divergence
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'
ARMS=('roadK8','match4','match4_shape_affine')
BANDS={'length_jsd':[(4.6,4.7),(4.8,4.9),(5.7,5.9)],'width_jsd':[(2.,2.1),(2.4,2.5)],
       'speed_jsd':[(0.,1.),(1.,5.),(5.,15.),(15.,30.)],'ang_dev_jsd':[(-5.,5.),(-10.,-5.),(5.,10.)]}


def main():
    observations=read(ART/'observations.json');diagnostic=read(ART/'histogram_diagnostics.json')
    assert diagnostic['observations_sha256']==sha(ART/'observations.json')
    records={};plot={}
    for arm in ARMS:
        rows=[r for r in observations if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,6144,4096)]
        assert len(rows)==2 and {r['eval_seed'] for r in rows}=={2817,3817}
        feature_rows={}
        for feature,bands in BANDS.items():
            hist=[read(Path(r['metrics_path']).parent/'feature_histograms.json')[feature] for r in rows]
            values=[divergence(h,100) for h in hist]
            assert hist[0]['matched_gt']==hist[1]['matched_gt']
            gt=values[0]['gt_mass'];generated=np.stack([v['generated_mass'] for v in values])
            contributions=np.stack([v['contributions'] for v in values])
            h=hist[0];centers=h['minimum']+(np.arange(len(gt))+.5)*h['bin_width']
            out=[]
            for lo,hi in bands:
                mask=(centers>=lo)&(centers<hi)
                out.append(dict(interval=[lo,hi],GT_mass=float(gt[mask].sum()),
                    generated_mass_per_GEN=[dict(GEN=r['eval_seed'],mass=float(p[mask].sum())) for r,p in zip(rows,generated)],
                    generated_mass_mean=float(generated[:,mask].sum(-1).mean()),
                    mean_jsd_contribution=float(contributions[:,mask].sum(-1).mean()),
                    fraction_of_mean_jsd=float(contributions[:,mask].sum(-1).mean()/contributions.sum(-1).mean())))
            feature_rows[feature]=dict(mean_jsd=float(contributions.sum(-1).mean()),bands=out)
            plot[arm,feature]=(centers,gt,generated)
        training=next(r['training_last256'] for r in diagnostic['results'] if (r['arm'],r['budget'],r['scenes'])==(arm,6144,4096))
        records[arm]=dict(features=feature_rows,training_last256=training)
    save(ART/'midpoint_attribute_diagnostic.json',dict(at_utc=now(),source_sha256=sha(__file__),
        observations_sha256=sha(ART/'observations.json'),TRAIN=4817,GEN=[2817,3817],updates=6144,scenes=4096,
        records=records,interpretation='Descriptive fixed-budget histogram intervals and actual training-window gradients. JSD contributions computed perGEN before averaging. No causal claim, no confidence intervals, no new candidate ranking.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    labels={'roadK8':'roadK8','match4':'matching4','match4_shape_affine':'matching4 + TRAIN affine size'}
    colors={'roadK8':'#646f79','match4':'#237bb8','match4_shape_affine':'#299c79'}
    features=['length_jsd','width_jsd','speed_jsd','ang_dev_jsd']
    ranges=[(3.5,7),(1.5,3.3),(0,32),(-30,30)]
    for ax,feature,limits in zip(axes.flat,features,ranges):
        for arm in ARMS:
            x,gt,generated=plot[arm,feature];error=(generated-gt)*100
            ax.plot(x,error.mean(0),color=colors[arm],lw=1.4,label=labels[arm])
            ax.fill_between(x,error.min(0),error.max(0),color=colors[arm],alpha=.10)
        ax.axhline(0,color='#9ba7ac',lw=.8);ax.set_xlim(*limits);ax.grid(alpha=.15)
        units='degrees' if feature=='ang_dev_jsd' else ('m/s' if feature=='speed_jsd' else 'm')
        ax.set_xlabel(feature.replace('_jsd','')+' ('+units+')');ax.set_ylabel('Generated minus GT (pp / bin)')
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.suptitle('6144 updates / TRAIN4817 / 4096 scenes / GEN2817 and3817',fontsize=12)
    fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,fontsize=9,frameon=False)
    fig.text(.5,.01,'Shading: two-GEN range, not confidence intervals. Native JSD uses all bins; common ranges shown.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.88));fig.savefig(ART/'midpoint_attribute_errors.png',dpi=170);plt.close(fig)
    print('MIDPOINT_ATTRIBUTE_DIAGNOSTIC_SAVED')


if __name__=='__main__':main()
