"""Plot native marginal errors for the completed four-way2048-step screen."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'
ARMS={'match4':('Matching4','#52768c'),
      'match4_shape_affine':('TRAIN-affine size','#138c77'),
      'match4_heading_p1':('Heading power1','#d19632'),
      'match4_affine_heading_p1':('Affine + heading power1','#ad5179')}
METRICS={'ang_dev_jsd':('Angle to lane (degrees)',(-40,40)),
         'length_jsd':('Length (m)',(3.4,6.5)),
         'width_jsd':('Width (m)',(1.7,2.9)),
         'speed_jsd':('Speed (m/s)',(0,30))}


def main():
    rows=read(ART/'observations.json');records=[]
    fig,axes=plt.subplots(2,2,figsize=(8,5.8));fig.subplots_adjust(top=.80,bottom=.19,hspace=.5,wspace=.3)
    provenance={};reference={}
    for arm,(label,color) in ARMS.items():
        members=sorted([r for r in rows if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,2048,4096)],key=lambda r:r['eval_seed'])
        assert [r['eval_seed'] for r in members]==[2817,3817]
        paths=[Path(r['metrics_path']).parent/'feature_histograms.json' for r in members]
        hs=[read(p) for p in paths];provenance.update({str(p):sha(p) for p in paths})
        for ax,(key,(xlabel,limits)) in zip(axes.flat,METRICS.items()):
            h=hs[0][key];q=np.array(h['matched_gt'],dtype=float);q/=q.sum()
            if key in reference:assert np.array_equal(reference[key],q)
            reference[key]=q
            generated=np.array([np.array(v[key]['generated'])/sum(v[key]['generated']) for v in hs])
            delta=100*(generated-q);x=h['minimum']+(np.arange(len(q))+.5)*h['bin_width']
            ax.plot(x,delta.mean(0),label=label,color=color,lw=1.6)
            ax.fill_between(x,delta.min(0),delta.max(0),color=color,alpha=.09)
            if key=='ang_dev_jsd':
                lo=round((-5-h['minimum'])/h['bin_width']);hi=round((5-h['minimum'])/h['bin_width'])
                records.append(dict(arm=arm,GEN=[2817,3817],within_5_degrees_generated=generated[:,lo:hi].sum(1).tolist(),
                    within_5_degrees_GT=float(q[lo:hi].sum()),per_GEN_jsd=[r[key] for r in members]))
            ax.set_xlim(*limits);ax.set_xlabel(xlabel,fontsize=9);ax.tick_params(labelsize=8.5)
            ax.grid(alpha=.15);ax.set_ylabel('Error per bin (percentage points)',fontsize=8)
    for ax in axes.flat:ax.axhline(0,color='#24333b',lw=.9,ls='--')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Marginal errors: 2048 updates / 4096 scenes / TRAIN4817',fontsize=11,y=.985)
    fig.text(.5,.025,'Lines: mean of two GEN histograms; shading: their range, not a confidence interval.\n'
             'Axes zoom into common values. Native per-GEN JSD uses all bins; these curves do not pool JSD.',
             ha='center',fontsize=8,color='#546574')
    path=ART/'affine_heading_marginal_errors.png';fig.savefig(path,dpi=180);plt.close(fig)
    save(ART/'affine_heading_marginal_errors.json',dict(at_utc=now(),source_sha256=sha(__file__),
        plot_sha256=sha(path),histogram_sha256=provenance,angle_peak_records=records,
        interpretation='Descriptive screen diagnostic. Generated marginal angle is relative to the nearest lane at generated position, not the supervised clean-heading reconstruction error. These plots do not isolate causality or predict full128epoch performance.'))
    print(records)


if __name__=='__main__':main()
