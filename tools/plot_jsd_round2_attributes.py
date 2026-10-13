"""Plot native attribute histograms; pooling is visual, never the R estimator."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_jsd_round2_plot')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'output/reports/init_diffusion_jsd_round2_20261013'


def main():
    folder=ART/'speed_isolation' if (ART/'speed_isolation/observations.json').exists() else ART
    rows=json.loads((folder/'observations.json').read_text())
    colors={'lane_agentmax05':'#0072B2','lane_attr_ot':'#D55E00','speed_only':'#009E73'}
    names={'lane_agentmax05':'Original lane','lane_attr_ot':'Speed + shape marginal loss','speed_only':'Speed-only marginal loss'}
    curves={}
    for arm in colors:
        cells=[r for r in rows if r['variant']==arm and r['seed']==5817 and r['budget']==4096]
        if len(cells)!=2:continue
        curves[arm]=[json.loads((Path(r['metrics_path']).parent/'feature_histograms.json').read_text()) for r in cells]
    assert curves
    figure,axes=plt.subplots(2,2,figsize=(12,6.8),dpi=200)
    for ax,(key,limits,title) in zip(axes.flat,[('speed_jsd',(1,8),'Speed 1-8 m/s (0-1 bin omitted)'),('speed_jsd',(8,25),'Higher speed'),('width_jsd',(1.7,2.7),'Width'),('length_jsd',(3,7),'Length')]):
        example=next(iter(curves.values()))[0][key]
        edges=example['minimum']+np.arange(len(example['matched_gt'])+1)*example['bin_width']
        visible=np.flatnonzero(((edges[:-1]+edges[1:])/2>=limits[0])&((edges[:-1]+edges[1:])/2<limits[1]))
        start,end=visible[0],visible[-1]+1
        q=np.array(example['matched_gt'],float);q/=q.sum()
        ax.stairs(q[start:end],edges[start:end+1],color='#222222',linewidth=1.6,label='Matched GT')
        for arm,hists in curves.items():
            g=sum((np.array(h[key]['generated'],float) for h in hists));g/=g.sum()
            ax.stairs(g[start:end],edges[start:end+1],color=colors[arm],linewidth=1.15,label=names[arm])
        ax.set_xlim(*limits);ax.set_title(title,fontsize=10)
        ax.set_xlabel('m/s' if key=='speed_jsd' else 'm');ax.set_ylabel('Probability per native bin')
        ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    figure.legend(handles,labels,ncol=len(labels),loc='upper center',bbox_to_anchor=(.5,.98),frameon=False,fontsize=9)
    figure.suptitle('TRAIN5817 / step4096 / 4096 scenes per GEN / two GEN pooled for visualization',y=1.01,fontsize=11)
    figure.tight_layout(rect=[0,0,1,.935])
    figure.savefig(ART/'attribute_histograms.png',bbox_inches='tight')
    plt.close(figure)
    print('Native attribute histograms saved:',list(curves))


if __name__=='__main__':main()
