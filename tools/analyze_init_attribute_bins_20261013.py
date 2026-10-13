"""Actual pooled full50K attribute histogram differences, without fitting."""
from pathlib import Path
import json
import numpy as np
from scipy.special import xlogy
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_evening_20261013'
read=lambda p:json.loads(Path(p).read_text())


def main():
    rows=read(BASE/'n20_sampler/observations.json')
    result={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.1))
    for ax,key in zip(axes,('speed_jsd','length_jsd','width_jsd')):
        summaries={}
        for arm in ('fresh657','freshK8'):
            rr=[r for r in rows if r['arm']==arm and r['scenes']==50000 and r['option']=='off_uniform']
            assert len(rr)==2
            bins=[read(Path(r['native_report']).parent/'feature_histograms.json')[key] for r in rr]
            real=np.sum([b['matched_gt'] for b in bins],0);gen=np.sum([b['generated'] for b in bins],0)
            p=gen/gen.sum();q=real/real.sum();mix=(p+q)/2
            p_ratio=np.divide(p,mix,out=np.ones_like(p),where=mix>0)
            q_ratio=np.divide(q,mix,out=np.ones_like(q),where=mix>0)
            contributions=.5*(xlogy(p,p_ratio)+xlogy(q,q_ratio))
            from scipy.spatial.distance import jensenshannon
            assert np.isclose(contributions.sum(),jensenshannon(gen,real)**2,atol=1.e-12)
            lo,step=bins[0]['minimum'],bins[0]['bin_width'];centers=lo+(np.arange(len(p))+.5)*step
            top=np.argsort(contributions)[-5:][::-1]
            summaries[arm]=dict(pooled_JSD_scaled=float(contributions.sum()*100),
                                generated_first_bin_mass=float(p[0]),GT_first_bin_mass=float(q[0]),
                                largest_bin_contributions=[dict(lower=float(lo+i*step),upper=float(lo+(i+1)*step),
                                                                  generated_mass=float(p[i]),GT_mass=float(q[i]),
                                                                  scaled_JSD_contribution=float(contributions[i]*100)) for i in top])
            ax.plot(centers,p,label=arm,linewidth=1.3)
            if arm=='fresh657':ax.plot(centers,q,label='matched GT',color='#333333',linewidth=1.2)
        ax.set_title(key);ax.set_ylabel('bin probability');ax.grid(alpha=.2)
        ax.set_xlim({'speed_jsd':(0,30),'length_jsd':(2.5,7),'width_jsd':(1.2,3)}[key])
        ax.set_xlabel('m/s' if key=='speed_jsd' else 'm')
        result[key]=summaries
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(BASE/'attribute_bins_full50000.png',dpi=190);plt.close(fig)
    (BASE/'attribute_bins_full50000.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
