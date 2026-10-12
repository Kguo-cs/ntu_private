"""Standalone plots from recorded paired observations; no training or metric changes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_init_diffusion_657_iteration import ART,RUNS,read,score

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    controls=read(ROOT/'logs/control657_paired_20261012/train4817/summary.json')
    stages=[('Round 1',RUNS/'medium')]
    stages.extend((p.name,p) for p in sorted(RUNS.glob('round2_medium_*')))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for title,folder in stages:
        if not (folder/'verified_completion.json').is_file():continue
        rows=read(folder/'summary.json')
        ax=axes[0 if title=='Round 1' else 1]
        for variant in sorted({r['variant'] for r in rows}):
            curve=[]
            for b in (2048,4096):
                selected=[r for r in rows if r['variant']==variant and r['budget']==b]
                if {r['eval_seed'] for r in selected}!={2817,3817}:continue
                s=score(selected,controls)
                curve.append((selected[0]['seen_training_examples']/493542,s['R'],s['gates_pass']))
            if not curve:continue
            ax.plot([x[0] for x in curve],[x[1] for x in curve],marker='o',label=variant)
            for e,r,gate in curve:
                if not gate:ax.scatter([e],[r],marker='x',color='#bc2626',s=80,zorder=10)
    for i,ax in enumerate(axes):
        ax.axhline(1,color='#303b48',ls='--',lw=1,label='budget-matched 657')
        ax.set(title=f'Round {i+1}: 4096 scenes, TRAIN4817, both GEN',xlabel='Equivalent training epochs',ylabel='Paired R (lower is better)')
        ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlim(1.9,4.5)
    fig.suptitle('Learning-curve reversals remain visible; red crosses fail at least one gate',fontsize=12)
    fig.savefig(ART/'paired_learning_curves.svg');fig.savefig(ART/'paired_learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    folders={'657':ROOT/'logs/control657_paired_20261012/train4817/control657'}
    folders.update({n:RUNS/'medium'/n for n in ('train_features','lane_centers','depth5')})
    recovery=RUNS/'medium_combo_recovery'
    if (recovery/'verified_completion.json').is_file():folders['adaptive_combo']=recovery/'adaptive_combo'
    for stage in sorted(RUNS.glob('round2_medium_*')):
        if (stage/'verified_completion.json').is_file():
            for row in read(stage/'summary.json'):
                if row['budget']==4096:folders[row['variant']]=stage/row['variant']
    gt_plotted=False
    for name,folder in folders.items():
        cells=[]
        for gen in (2817,3817):
            path=folder/'eval_step_4096'
            if gen==3817:path=path/f'eval_seed_{gen}'
            path=path/'ema/feature_histograms.json'
            if path.is_file():cells.append(read(path))
        if len(cells)!=2:continue
        for ax,key,limits in zip(axes,('length_jsd','width_jsd'),((3,7),(1.4,2.8))):
            v=cells[0][key];xx=v['minimum']+(np.arange(len(v['generated']))+.5)*v['bin_width']
            probability=np.mean([np.asarray(c[key]['generated'])/sum(c[key]['generated']) for c in cells],axis=0)
            ax.step(xx,probability*100,where='mid',label=name,alpha=.85)
            if not gt_plotted:
                target=np.asarray(v['matched_gt'])/sum(v['matched_gt'])
                ax.step(xx,target*100,where='mid',label='GT',color='black',lw=2)
            ax.set(xlim=limits,xlabel=key.removesuffix('_jsd')+' (m)',ylabel='Probability per 0.1 m bin (%)')
            ax.grid(alpha=.2)
        gt_plotted=True
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('Shape distribution at 4.25 epochs: same 4096 scenes, GEN mean',fontsize=12)
    fig.savefig(ART/'shape_histograms.svg');fig.savefig(ART/'shape_histograms.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    long_rows=[]
    for folder in sorted(RUNS.glob('round2_long*')):
        if (folder/'verified_completion.json').is_file():long_rows.extend(read(folder/'summary.json'))
    controls_long=[]
    for seed in (4817,5817):controls_long.extend(read(ROOT/f'logs/control657_paired_20261012/train{seed}/summary.json'))
    for variant in sorted({r['variant'] for r in long_rows}):
        for seed in sorted({r['seed'] for r in long_rows if r['variant']==variant}):
            curve=[]
            for budget in (3413,6826,10240):
                selected=[r for r in long_rows if r['variant']==variant and r['seed']==seed and r['budget']==budget]
                if {r['eval_seed'] for r in selected}!={2817,3817}:continue
                score_row=score(selected,controls_long)
                curve.append((selected[0]['seen_training_examples']/493542,score_row))
            for ax,key in zip(axes,('R','R_attr')):
                if not curve:continue
                ax.plot([x[0] for x in curve],[x[1][key] for x in curve],marker='o',label=f'{variant}, TRAIN{seed}')
                for epoch,result in curve:
                    if not result['gates_pass']:ax.scatter([epoch],[result[key]],marker='x',s=80,color='#bc2626')
    for ax,key in zip(axes,('R','R_attr')):
        ax.axhline(1,color='#303b48',ls='--',lw=1,label='budget-matched 657')
        ax.set(xlabel='Equivalent training epochs',ylabel=key+' (lower is better)',xlim=(3,11))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Longer confirmation: 4096 matched scenes, mean of both GEN; verified stages only',fontsize=12)
    fig.savefig(ART/'long_confirmation_curves.svg');fig.savefig(ART/'long_confirmation_curves.png',dpi=160);plt.close(fig)
    print('Saved paired, shape and longer-confirmation plots')

if __name__=='__main__':main()
