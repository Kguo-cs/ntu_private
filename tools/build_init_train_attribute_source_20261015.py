"""Fit an immutable empiricalattribute source using TRAIN cache fields only."""
from pathlib import Path
import os,gc,argparse,hashlib,json,pickle,sys,datetime
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
gc.disable()
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch);torch.set_num_threads(1)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenes',type=int,default=16384);p.add_argument('--seed',type=int,default=241015);p.add_argument('--knots',type=int,default=2049);p.add_argument('--artifact-name',default='init_diffusion_train_attribute_source_20261015');a=p.parse_args()
 assert a.artifact_name==Path(a.artifact_name).name and (a.knots==0 or a.knots>=3)
 source=ROOT/'src/waymo_data/full/training_map2_sd';catalogue=ROOT/'src/waymo_data/full/training_map2_sd_files.pkl'
 names=pickle.loads(catalogue.read_bytes())['files'];assert len(names)==len(set(names))==493542 and 1<=a.scenes<=len(names)
 ids=np.random.default_rng(a.seed).choice(len(names),size=a.scenes,replace=False);selected=[names[int(i)] for i in ids]
 out=ROOT/'docs/experiments'/a.artifact_name;out.mkdir(exist_ok=False)
 selection=out/'TRAIN_files.pkl';selection.write_bytes(pickle.dumps(dict(files=selected)))
 columns=[[],[],[]];invalid=[0,0,0];total=0;scenes=0;over40=over80=zero=slow=0
 for name in selected:
  assert name==Path(name).name
  data=torch.load(source/name,map_location='cpu',weights_only=False);agent=data['tokenized_agent'];shape=agent.get('initial_shape',agent.get('shape'));velocity=agent['local_vel'];assert shape.ndim==velocity.ndim==2 and shape.shape[0]==velocity.shape[0] and velocity.shape[1]==2
  fields=[shape[:,0].float(),shape[:,1].float(),velocity.float().norm(dim=-1)];total+=len(shape)
  for j,x in enumerate(fields):
   mask=torch.isfinite(x)&((x>0) if j<2 else (x>=0));invalid[j]+=int((~mask).sum());columns[j].append(x[mask].clone())
  valid_speed=fields[2][torch.isfinite(fields[2])&(fields[2]>=0)];zero+=int((valid_speed==0).sum());slow+=int((valid_speed<.05).sum());over40+=int((valid_speed>40).sum());over80+=int((valid_speed>80).sum());scenes+=1
  del data,agent,shape,velocity,fields,valid_speed
  if scenes%2048==0:print('TRAIN_ATTRIBUTE_SOURCE',scenes,flush=True)
 values=[torch.cat(v) for v in columns];knots=a.knots or max(len(v) for v in values);ranks=torch.linspace(0,1,knots)
 table=torch.stack([v.sort().values if len(v)==knots and a.knots==0 else torch.quantile(v,ranks) for v in values],-1)
 assert table.shape==(knots,3) and torch.isfinite(table).all() and (table[1:]>=table[:-1]).all() and (table[:,:2]>0).all() and (table[:,2]>=0).all()
 payload=dict(kind='quantile_table',fields=['length','width','speed'],quantile_values=table,TRAIN_scenes=scenes,TRAIN_selection_sha256=sha(selection),role='unconditional TRAIN marginal overall agenttypes includingego; no GTtype/attributeconditioning atgeneration')
 target=out/'quantiles.pt';torch.save(payload,target)
 stat=dict(status='TRAIN_ONLY_ATTRIBUTE_QUANTILES_READY_NO_MODEL_TRAINED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(source),catalogue_sha256=sha(catalogue),selection_sha256=sha(selection),selection_seed=a.seed,TRAIN_scenes=scenes,total_agent_rows=total,invalid_by_field=invalid,valid_by_field=[len(v) for v in values],mean=[float(v.mean()) for v in values],std=[float(v.std(unbiased=False)) for v in values],min=[float(v.min()) for v in values],max=[float(v.max()) for v in values],speed_zero_fraction=zero/len(values[2]),speed_under005_fraction=slow/len(values[2]),speed_over40_count=over40,speed_over80_count=over80,quantiles=str(target),quantiles_sha256=sha(target),knots=knots,full_sorted_table=a.knots==0,source_code_sha256=sha(Path(__file__)),no_validation_data=True,no_optimization=True,no_new_parameters=True,notes='Each marginal uses its own validTRAIN observations; invalidsize rows do not remove validspeeds. No clipping of rare TRAIN endpoints. Generation willtransform existing Gaussian draws, not read GT attributes.')
 (out/'provenance.json').write_text(json.dumps(stat,indent=2)+'\n');assert not torch.cuda.is_initialized();print(json.dumps(stat),flush=True);os._exit(0)
if __name__=='__main__':main()
