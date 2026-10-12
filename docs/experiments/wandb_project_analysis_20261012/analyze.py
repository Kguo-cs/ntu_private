"""Reproducible project-level analysis of frozen W&B snapshots."""
import ast
import collections
import datetime
import json
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
inventory=json.loads((ROOT/'inventory.json').read_text())
METRICS=['nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate']
SCALE={'nearest_dist_jsd':10,'lat_dev_jsd':10,'ang_dev_jsd':100,'length_jsd':100,'width_jsd':100,'speed_jsd':100,'collision_rate':1}
PRIMARY=['iu3fa4gx','vd23g1bq','0yhfg9ue','pz8g135r','4bedivn1','pd7zg04g']
RECENT=['v62ymyjj','fs0kq6fc','c486qs4q']
NAMES={'iu3fa4gx':'657 loss_search / B512','vd23g1bq':'frozen shared / B512','0yhfg9ue':'frozen shared / B256','pz8g135r':'overnight separate / B256','4bedivn1':'scratch / B512 (run0)','pd7zg04g':'scratch / B512 (run3)','v62ymyjj':'lr 2e-4, floor .01','fs0kq6fc':'lr 2e-4, floor .05','c486qs4q':'lr 5e-4, floor .01'}
COLORS=dict(zip(PRIMARY,['#2a6fbb','#b65d14','#4b9e63','#ba498a','#7461a8','#789299']))
records={r['id']:r for r in inventory['runs']}
H={}; V={}; manifests=[]
all_keys=set(); total=0;groups=collections.Counter()
for rid,r in records.items():
 h=json.loads((ROOT/'histories'/f'{rid}.json').read_text())['rows'];H[rid]=h; total+=len(h)
 for row in h:all_keys.update(row)
 V[rid]=[a for a in h if all(isinstance(a.get(k),(int,float)) and math.isfinite(a[k]) for k in METRICS)]
 c=r['config'];m=c.get('model',{}).get('model_config',c.get('model_config',{}));m=ast.literal_eval(m) if isinstance(m,str) else m
 d=m.get('decoder',{});i=d.get('init_diffusion') or {};data=c.get('data',{});t=c.get('trainer',{})
 groups[(d.get('init_decoder'),m.get('sd_reference_mode','unspecified'))]+=1
 meta_path=ROOT/'metadata'/rid/'selected.json';meta=json.loads(meta_path.read_text()) if meta_path.exists() else None
 manifests.append({'id':rid,'name':r['name'],'state':r['state'],'commit':r['commit'],'url':r['url'],
  'seed':c.get('seed'),'train_batch_size':data.get('train_batch_size'),'val_batch_size':data.get('val_batch_size'),
  'limit_val_batches':t.get('limit_val_batches'),'train_data':data.get('train_raw_dir'),'validation_data':data.get('val_raw_dir'),
  'init_decoder':d.get('init_decoder'),'reference_mode':m.get('sd_reference_mode'),'lr':m.get('lr'),'lr_min_ratio':m.get('lr_min_ratio'),
  'precision':t.get('precision'),'trainer_devices':t.get('devices'),'matmul_precision_recorded_in_config':c.get('float32_matmul_precision'),
  'initial_map_checkpoint':m.get('initial_map_checkpoint'),'ckpt_path':c.get('ckpt_path'),'crop':m.get('token_processor',{}).get('init_map_crop'),
  'map_half_extent':m.get('token_processor',{}).get('init_map_half_extent'),'init_diffusion_config':i,
  'validation_rows':len(V[rid]),'first_validation_epoch':V[rid][0].get('epoch') if V[rid] else None,
  'last_validation_epoch':V[rid][-1].get('epoch') if V[rid] else None,'last_training_epoch_from_summary':r['summary'].get('epoch'),'execution_metadata':meta})

def means(rid,epochs):
 rows=[a for a in V[rid] if a.get('epoch') in epochs]
 return {k:statistics.mean(a[k] for a in rows) for k in METRICS} if rows else None

analysis={'inventory_fetched_at_utc':inventory.get('fetched_at_utc'),'analyzed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'run_count':len(records),'history_rows':total,'validation_rows':sum(map(len,V.values())),
 'states':dict(collections.Counter(r['state'] for r in records.values())),
 'groups':[{'initializer':a,'reference_mode':b,'count':n} for (a,b),n in groups.items()],
 'metric_multipliers':SCALE,'primary_at_epoch55':{},'primary_mean_epochs47_51_55':{},'drift':{},'ratios_mean47_51_55':[],
 'recent_early_metrics':{r:[{k:a.get(k) for k in ['epoch','trainer/global_step']+METRICS} for a in V[r]] for r in RECENT},
 'limitations':['Most online validation uses only first 256 scenes; directory membership hashes are unavailable for remote machines.',
 'Full reference and matched reference are distinct cohorts and must not be ranked together.',
 'Recipes change seed, initialization, batch, normalization, loss and sometimes entrypoint/hardware together; associations are not single-variable causal effects.',
 'No onroad/offroad ratio, generated vehicle/type counts or attribute histograms were logged.',
 'Each component minimum can occur at a different checkpoint. No composite checkpoint is formed from these minima.',
 'GPU inventory count is not trainer device count. Metadata shows four GPUs on one host, but these configs specify trainer.devices=1.',
 'Training loss fractions are not gradient fractions.',
 'Matmul precision and remote checkpoint/data hashes were not logged in run config.']}
for rid in PRIMARY:
 row=next(a for a in V[rid] if a.get('epoch')==55)
 analysis['primary_at_epoch55'][rid]={k:row[k] for k in ['epoch','trainer/global_step']+METRICS}
 analysis['primary_mean_epochs47_51_55'][rid]=means(rid,[47,51,55])
 drift={}
 for k in METRICS:
  a=min(V[rid],key=lambda z:z[k]);b=V[rid][-1]
  drift[k]={'minimum':a[k],'minimum_epoch':a['epoch'],'last':b[k],'last_epoch':b['epoch'],'rise_percent':100*(b[k]/a[k]-1)}
 for lo,hi in [(32,63),(96,127)]:
  rows=[a for a in H[rid] if lo<=a.get('epoch',-1)<=hi]
  drift[f'train_mean_epoch{lo}_{hi}']={k:statistics.mean(a[k] for a in rows if isinstance(a.get(k),(int,float))) for k in ['train/loss','train/component/velocity_loss','train/component/width_loss'] if any(isinstance(a.get(k),(int,float)) for a in rows)}
 analysis['drift'][rid]=drift
for a,b in [('pz8g135r','vd23g1bq'),('pz8g135r','0yhfg9ue'),('vd23g1bq','iu3fa4gx'),('vd23g1bq','4bedivn1'),('0yhfg9ue','vd23g1bq')]:
 am=means(a,[47,51,55]);bm=means(b,[47,51,55])
 analysis['ratios_mean47_51_55'].append({'candidate':a,'baseline':b,'percent_change':{k:100*(am[k]/bm[k]-1) for k in METRICS}})
(ROOT/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False))
(ROOT/'run_manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2,allow_nan=False))
(ROOT/'validation_history.json').write_text(json.dumps({r:[{k:a.get(k) for k in ['epoch','trainer/global_step','_step','_timestamp']+METRICS} for a in v] for r,v in V.items()},ensure_ascii=False,allow_nan=False))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
for ax,k in zip(axes.flat,METRICS):
 for rid in PRIMARY:
  rows=V[rid];ax.plot([a['epoch'] for a in rows],[a[k] for a in rows],label=NAMES[rid],color=COLORS[rid],lw=1.8,alpha=.9)
 ax.axvline(55,color='#777',ls=':',lw=1)
 ax.set(title=k,xlabel='Validation epoch (zero based)',ylabel='Native metric value');ax.grid(alpha=.18)
 if k in ['nearest_dist_jsd','lat_dev_jsd']:ax.set_yscale('log')
axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='upper left',frameon=False,title='Matched reference, GT maps, EMA')
axes.flat[-1].text(.04,.18,'First 256 validation scenes.\nBatch, seed, loss and hardware differ.\nDescriptive curves, not causal ablations.\nEach marker-free line uses all logged\nvalidation points, not sampled history.',transform=axes.flat[-1].transAxes,fontsize=10)
fig.suptitle('Online long-training curves: geometry / collision / attributes trade off',fontsize=16)
fig.savefig(ROOT/'long_training_curves.png',dpi=160);fig.savefig(ROOT/'long_training_curves.svg');plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained');rid='iu3fa4gx'
for ax,k in zip(axes[:2],['speed_jsd','width_jsd']):
 rows=V[rid];ax.plot([a['epoch'] for a in rows],[a[k] for a in rows],color=COLORS[rid],lw=2)
 a=min(rows,key=lambda a:a[k]);ax.scatter([a['epoch'],rows[-1]['epoch']],[a[k],rows[-1][k]],color=['#2b8b57','#bd4d43'],zorder=3)
 ax.annotate(f"epoch {a['epoch']}: {a[k]:.4f}",(a['epoch'],a[k]),xytext=(0,14),textcoords='offset points',ha='center',fontsize=9)
 ax.set(title=k,xlabel='Validation epoch',ylabel='Native metric value');ax.grid(alpha=.2)
ax=axes[-1];rows=[a for a in H[rid] if 'train/component/velocity_loss' in a]
for k,color in [('train/component/velocity_loss','#b65d14'),('train/component/width_loss','#7461a8')]:
 bins={}
 for a in rows:
  if isinstance(a.get(k),(int,float)):bins.setdefault(a['epoch'],[]).append(a[k])
 xs=sorted(bins);ax.plot(xs,[statistics.mean(bins[x]) for x in xs],label=k.replace('train/component/',''),lw=2,color=color)
ax.set(title='Training reconstruction keeps improving',xlabel='Training epoch',ylabel='Epoch mean component loss');ax.legend(frameon=False);ax.grid(alpha=.2)
fig.suptitle('657 loss_search: lower reconstruction loss does not ensure lower generation JSD',fontsize=14)
fig.savefig(ROOT/'attribute_drift.png',dpi=160);fig.savefig(ROOT/'attribute_drift.svg');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,k in zip(axes.flat,['lat_dev_jsd','speed_jsd','width_jsd','collision_rate']):
 for j,rid in enumerate(['iu3fa4gx']+RECENT):
  rows=[a for a in V[rid] if a['epoch']<=11]
  label='lr 5e-4, floor .05 (657)' if rid=='iu3fa4gx' else NAMES[rid]
  ax.plot([a['epoch'] for a in rows],[a[k] for a in rows],marker='o',label=label)
 ax.set(title=k,xlabel='Validation epoch',ylabel='Native metric value');ax.grid(alpha=.2)
axes.flat[0].legend(frameon=False,fontsize=8)
fig.suptitle('New LR runs are early; 657 also differs in host / entrypoint / code revision',fontsize=13)
fig.savefig(ROOT/'early_learning_rate.png',dpi=160);fig.savefig(ROOT/'early_learning_rate.svg');plt.close(fig)
print(json.dumps({k:analysis[k] for k in ['run_count','history_rows','validation_rows','states','groups']},ensure_ascii=False))
print('Saved analysis.json, run_manifest.json, validation_history.json and 3 PNG/SVG plots.')
