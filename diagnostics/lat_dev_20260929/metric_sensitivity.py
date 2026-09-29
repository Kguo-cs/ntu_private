import ast, json, sqlite3, pickle, zlib, math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree, distance
from collections import defaultdict

ROOT=Path('/home/ke/code/sim'); OUT=Path('/tmp/sim_lat_audit')
def functions(path):
 tree=ast.parse(Path(path).read_text()); keep=[]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in ('get_onroad_vehicles','get_lateral_devs','jsd'): keep.append(node)
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='UNIFIED_FORMAT_INDICES' for t in node.targets):keep.append(node)
 ns={'np':np,'distance':distance,'UNIFIED_FORMAT_INDICES':{'pos_x':0,'pos_y':1,'speed':2,'cos_heading':3,'sin_heading':4,'length':5,'width':6}};exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),ns);return ns
old=functions(ROOT/'src/smart/metrics/old_gen_metrics.py')
off=functions('/home/ke/code/scenario-dreamer/utils/metrics_helpers.py')
c=sqlite3.connect((ROOT/'src/waymo_data/sd_real_metric_cache.sqlite').as_uri()+'?mode=ro',uri=True)
md={k:json.loads(v) for k,v in c.execute('select key,value from metadata')}
vals=defaultdict(list); total=0; exact=True; ego_errors=[]; step_lengths=[]
bins=np.arange(0,1.6,.1); n=512
for ordinal,name,payload in c.execute('select ordinal,name,payload from scenes order by ordinal limit ?', (n,)):
 entry=pickle.loads(zlib.decompress(payload));gt=entry['gt'];v=np.asarray(gt['vehicles']);lanes=np.asarray(gt['metric_lanes']);p=v[:,:2];total+=len(v)
 if not len(v):continue
 tree=cKDTree(lanes.reshape(-1,2));ds,idx=tree.query(p); vals['gt'].append(ds)
 if ordinal<32:
  aa=old['get_lateral_devs'](old['get_onroad_vehicles'](v,lanes),lanes)
  bb=off['get_lateral_devs'](off['get_onroad_vehicles'](v,lanes),lanes)
  exact=exact and np.array_equal(aa,bb)
 # Nearest polyline point to obtain local lane normal
 li=idx//100;pi=np.clip(idx%100,1,99);t=lanes[li,pi]-lanes[li,pi-1];t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12);normal=np.column_stack([-t[:,1],t[:,0]])
 for shift in [.05,.1,.2,.5,1.,2.]:
  vals['shift_'+str(shift)].append(tree.query(p+shift*normal)[0])
 # Exact point-to-segment distance + projected position (quantization floor)
 a=lanes[:,:-1].reshape(-1,2);b=lanes[:,1:].reshape(-1,2);u=b-a;u2=np.sum(u*u,axis=1)
 w=p[:,None]-a[None];alpha=np.clip(np.sum(w*u[None],axis=-1)/np.maximum(u2[None],1e-12),0,1)
 proj=a[None]+alpha[...,None]*u[None];segds=np.linalg.norm(p[:,None]-proj,axis=-1);si=segds.argmin(axis=1)
 vals['exact_segment'].append(segds[np.arange(len(p)),si]);vals['snap_segment'].append(tree.query(proj[np.arange(len(p)),si])[0]);vals['snap_point'].append(np.zeros(len(p)))
 step_lengths.extend(np.sqrt(u2[u2>1e-12]).tolist())
 # Preserve last-agent ego contract and identify its reference vehicle row geometrically.
 f=ROOT/'src/waymo_data/full/scenario_dreamer_val'/f'{ordinal:05d}_{Path(name).stem}.pt'
 d=torch.load(f,map_location='cpu',weights_only=False);info=d['scenario_dreamer'];raw=d['agent'];ts=int(d['scene_timestep'])
 ep=np.asarray(raw['position'][-1,ts,:2],dtype=np.float64)-np.asarray(info['center_world'],dtype=np.float64);ang=float(info['rotation_angle']);co,si=np.cos(ang),np.sin(ang);ep=np.array([co*ep[0]-si*ep[1],si*ep[0]+co*ep[1]])
 err=np.linalg.norm(p-ep,axis=1);ei=int(err.argmin());ego_errors.append(float(err[ei]))
 if err[ei]<.05:
  vals['ego'].append(ds[ei:ei+1]);vals['non_ego'].append(np.delete(ds,ei))
 else: vals['non_ego'].append(ds)
vals={k:np.concatenate(v) for k,v in vals.items()};base=vals['gt'];base=base[base<=1.5]
def stat(x):
 on=x[x<=1.5]; h=np.histogram(on,bins=bins)[0]; q=np.histogram(base,bins=bins)[0]
 return {'agents':len(x),'onroad_fraction':float(np.mean(x<=1.5)),'mean_distance_all':float(np.mean(x)),'mean_distance_onroad':float(np.mean(on)),'lat_jsd_matched':float(distance.jensenshannon(h,q)**2*10),'hist_onroad':h.tolist()}
result={'scenes':n,'total_vehicles':total,'old_official_exact_equal_first32':exact,'ego_matching_error_max':max(ego_errors),'lane_point_spacing_quantiles_m':np.quantile(step_lengths,[.1,.5,.9,.99]).tolist(),'experiments':{k:stat(v) for k,v in vals.items()}}
# Removing arbitrary whole repeated scene copies has zero effect on conditional JSD.
result['blindspot_duplicate_gt_then_move_one_copy_1000m']={'lat_jsd_matched':0.,'onroad_fraction':len(base)/(2*total)}
(OUT/'metric_sensitivity.json').write_text(json.dumps(result,indent=2))
print(json.dumps({**result,'experiments':{k:{kk:vv for kk,vv in s.items() if kk!='hist_onroad'} for k,s in result['experiments'].items()}},indent=2))
