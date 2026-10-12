"""CPU-only plotting validation with explicit synthetic layout/statistic fixtures."""
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, '/home/ke/code/sim')
from tools import plot_frozen_map_search as p
from tools.analyze_frozen_map_search import exposure
p.plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'svg.fonttype':'none'})
qa=Path('/tmp/sim_frozen_map_long_curve_cpu_qa');qa.mkdir(exist_ok=True)
(qa/'FIXTURES_ONLY_NOT_SCIENTIFIC_RESULTS.txt').write_text('Synthetic CPU verification values only. Never include these PNGs or values in report search results.\n')
actual,w,_=p.discover([p.ROOT/'logs/component_loss_frozen_map'])
actual_long=[row for row in actual if Path(row['run_path']).name.startswith(('long_','confirm_'))]
template=copy.deepcopy(actual_long[0])
methods={'new_control':(512,1.0), 'combo_ego_sincos_batch256':(256,.8), 'ego_sincos':(512,1.1)}
rows=[]
for variant,(batch,factor) in methods.items():
 for checkpoint_index,budget512 in enumerate((1024,2048,3072,4096)):
  budget=budget512*(512//batch)
  seeds=(817,) if checkpoint_index==0 or variant=='ego_sincos' else (817,1817,2817)
  for seed_index,seed in enumerate(seeds):
   folder=qa/(('long_seed817_' if seed==817 else 'confirm_seed'+str(seed)+'_')+variant);folder.mkdir(exist_ok=True)
   (folder/'protocol.json').write_text(json.dumps({'eval_seed':2817,'extra_eval_seed':3817})+'\n')
   for gen_index,gen in enumerate((2817,3817)):
    row=copy.deepcopy(template);value=factor*(.02*(checkpoint_index+1)+seed_index*.1+gen_index*.02)
    row.update(run_path=str(folder),summary_source=str(folder/'summary.json'),variant=variant,arm_id=variant+'@cpu_fixture',arm_config_sha256=variant,seed=seed,eval_seed=gen,budget=budget,train_batch_size=batch,seen_training_examples=exposure(budget,batch,493542),cohort_sha256='FIXTURE_ONLY_NOT_RESULT')
    for metric in p.METRICS:row[metric]=value*p.MULTIPLIERS[metric]
    rows.append(row)
captured={}
original=p.save_figure
def capture(fig,output,name):
 if name=='long_confirmation_speed_jsd':
  ax=fig.axes[0]
  captured['mean_lines']=[{'label':line.get_label(),'x':list(map(float,line.get_xdata())),'y':list(map(float,line.get_ydata()))} for line in ax.lines if not line.get_label().startswith('_')]
  captured['error_segments']=[segment.tolist() for collection in ax.collections if hasattr(collection,'get_segments') for segment in collection.get_segments()]
 return original(fig,output,name)
p.save_figure=capture
info=p.long_curves(rows,qa,'ema',493542)
p.save_figure=original
curves=info['cohorts'][0]['curves']
expected_exp=[524262,1048524,1572786,2097048]
for variant,(batch,factor) in methods.items():
 points=sorted(curves[variant+'@cpu_fixture'],key=lambda x:x['seen_training_examples'])
 assert [point['seen_training_examples'] for point in points]==expected_exp
 assert [point['x_observed_training_epochs'] for point in points]==[value/493542 for value in expected_exp]
 expected_n=[1,1,1,1] if variant=='ego_sincos' else [1,3,3,3]
 assert [point['statistics']['num_independent_training_seeds'] for point in points]==expected_n
 for index,point in enumerate(points):
  stat=point['statistics'];n=expected_n[index]
  assert stat['num_actual_observations']==2*n
  assert all(value==[2817,3817] for value in stat['generation_seeds_by_training_seed'].values())
  expected_mean=factor*(.02*(index+1)+.01+(.1 if n==3 else 0))
  expected_sd=factor*(.1 if n==3 else 0)
  for metric in p.METRICS:
   got=stat['statistics_native'][metric];scale=p.MULTIPLIERS[metric]
   assert got['n']==n
   assert abs(got['mean']/scale-expected_mean)<1e-12
   assert abs(got['sd']/scale-expected_sd)<1e-12
  line=next(line for line in captured['mean_lines'] if line['label'].startswith(variant.replace('_',' ').split(' batch256')[0]))
  assert line['x'][index]==expected_exp[index]/493542
  assert abs(line['y'][index]-expected_mean)<1e-12
assert len(captured['error_segments'])==6
assert all(abs(segment[0][0]-expected_exp[0]/493542)>1e-12 for segment in captured['error_segments'])
assert sorted(round(segment[1][1]-segment[0][1],12) for segment in captured['error_segments'])==[.16,.16,.16,.2,.2,.2]
assert 'B256; TRAIN N=1,3' in str(captured['mean_lines'])
assert 'B512; TRAIN N=1' in str(captured['mean_lines'])
partial=[row for row in rows if not (row['variant']=='new_control' and row['budget']==2048 and row['seed']==2817 and row['eval_seed']==3817)]
partialdir=qa/'incomplete_one_GEN';partialdir.mkdir(exist_ok=True)
partial_result=p.long_curves(partial,partialdir,'ema',493542)
assert len(partial_result['skipped_incomplete_generation_cells'])==1
point=next(point for point in partial_result['cohorts'][0]['curves']['new_control@cpu_fixture'] if point['seen_training_examples']==1048524)
assert point['statistics']['num_independent_training_seeds']==2
assert abs(point['statistics']['statistics_native']['speed_jsd']['mean']/100-.10)<1e-12
assert abs(point['statistics']['statistics_native']['speed_jsd']['sd']/100-statistics.stdev([.05,.15]))<1e-12
incomplete_all=[row for row in rows if row['variant']=='new_control' and row['eval_seed']==2817]
skipped=p.long_curves(incomplete_all,qa,'ema',493542)
assert skipped['status']=='skipped' and not skipped['files']
duplicate=copy.deepcopy(rows[0]);duplicate['speed_jsd']+=1
try:p.long_curves(rows+[duplicate],qa,'ema',493542)
except ValueError as error:duplicate_rejected=str(error)
else:raise AssertionError('Conflicting duplicate cell accepted')
bad_method=copy.deepcopy(rows[0]);bad_method.update(seed=3817,eval_seed=2817,train_batch_size=256,budget=2048)
try:p.long_curves(rows+[bad_method],qa,'ema',493542)
except ValueError as error:method_rejected=str(error)
else:raise AssertionError('Conflicting method accepted')
assert all(layout['minimum_source_font_pt']>=9 and layout['minimum_effective_font_at_report_box_pt']>=8 and not layout['nonempty_visible_text_outside_canvas'] for layout in p.FIGURE_LAYOUTS.values())
assert 'torch' not in sys.modules
result={'scope':'CPU_FIXTURES_ONLY_NOT_SCIENTIFIC_RESULTS','actual_available_long_observations':len(actual_long),'actual_available_variants':sorted({row['variant'] for row in actual_long}),'actual_discovery_warnings':w,'fixture_rows':len(rows),'exposure_pairs':[{'B512_updates':a,'B256_updates':2*a,'actual_exposure':exposure(a,512,493542),'matching_B256_actual_exposure':exposure(2*a,256,493542),'equivalent_full_epochs':exposure(a,512,493542)/493542,'not_naive_steps_times_batch':exposure(a,512,493542)!=a*512} for a in (1024,2048,3072,4096)],'assertions':'PASS: two GEN repetitions averaged inside TRAIN; TRAIN N=1/3, not2/6; across-TRAIN sample SD; plotted raw JSD/scaled native SD; only N>1 errorbars; legends observed batch/N; exact equal-exposure x; incomplete planned GEN cells skipped; conflicting cells/methods rejected.','captured_speed_plot':captured,'skipped_incomplete_GEN_example':partial_result['skipped_incomplete_generation_cells'],'duplicate_rejected':duplicate_rejected,'method_rejected':method_rejected,'layouts':p.FIGURE_LAYOUTS,'validation_source':str(Path(__file__).resolve()),'validation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plot_tool_sha256':p.sha256(p.__file__),'torch_imported':False}
(qa/'cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'scope':result['scope'],'status':'PASS','exposure_pairs':result['exposure_pairs'],'rendered_metric_figure_count':len(p.FIGURE_LAYOUTS),'minimum_report_fit_font_pt':min(x['minimum_effective_font_at_report_box_pt'] for x in p.FIGURE_LAYOUTS.values()),'captured_speed_errorbar_count':len(captured['error_segments']),'torch_imported':False},indent=2))
