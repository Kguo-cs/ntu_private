"""CPU-only audit of local Scenario Dreamer samples; writes diagnostic JSON only."""
import argparse
import json, random, collections
from pathlib import Path
import numpy as np
import torch
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[2] / 'src/waymo_data')
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('data_stats.json'))
parser.add_argument('--samples', type=int, default=1024)
parser.add_argument('--seed', type=int, default=817)
args = parser.parse_args()
ROOT=args.data_root
SEED=args.seed
N=args.samples
def arr(v): return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
def quant(v):
 a=np.asarray(v); return {'min':float(a.min()),'p50':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max()),'mean':float(a.mean())}
def local(v,h):
 return np.stack([v[...,0]*np.cos(h)+v[...,1]*np.sin(h),-v[...,0]*np.sin(h)+v[...,1]*np.cos(h)],axis=-1)
out={'seed':SEED,'sample_size_per_split':N}
allrows={}
trainpaths=sorted((ROOT/'full/training_map2_sd').glob('*.pt'))
sids=[p.stem.rsplit('_',2)[0] for p in trainpaths]
times=[int(p.stem.rsplit('_',1)[1]) for p in trainpaths]
c=collections.Counter(sids)
out['train_inventory']={'files':len(trainpaths),'unique_scenario_ids':len(c),'scenarios_with_multiple_files':sum(v>1 for v in c.values()),'max_files_per_scenario':max(c.values()),'reference_time_counts':dict(sorted(collections.Counter(times).items()))}
for split,paths in [('train',trainpaths),('val',sorted((ROOT/'full/scenario_dreamer_val').glob('*.pt')))]:
 sample=random.Random(SEED).sample(paths,min(N,len(paths)))
 stats=collections.Counter(); nums=[]; maps=[]; rows=[]; errors=collections.defaultdict(list); scene_times=[]
 for p in sample:
  d=torch.load(p,map_location='cpu',weights_only=False)
  if split=='train':
   a=d['tokenized_agent']; pos=arr(a['initial_pos']).astype(float); head=arr(a['initial_heading']).astype(float); vel=arr(a['local_vel']).astype(float); shape=arr(a['shape']).astype(float); typ=arr(a['type']); maps.append(len(d['tokenized_map']['position'])); scene_times.append(int(p.stem.rsplit('_',1)[1]))
  else:
   a=d['agent']; t=int(d['scene_timestep']); pos=arr(a['position'][:,t,:2]).astype(float); head=arr(a['heading'][:,t]).astype(float); vel=local(arr(a['velocity'][:,t]).astype(float),head); shape=arr(a['shape']).astype(float); typ=arr(a['type']); maps.append(len(d['map_save']['traj_pos'])); scene_times.append(t)
   stats['invalid_agents_at_reference']+=int((~arr(a['valid_mask'][:,t])).sum()); stats['incorrect_last_ego']+=int(not bool(a['role'][-1,0])); stats['bad_ego_count']+=int(a['role'][:,0].sum()!=1)
   s=d['scenario_dreamer']; sd=arr(s['agent_states']); ids=arr(s['source_index']); outputids=arr(s['output_source_index']); idx=np.array([np.flatnonzero(outputids==i)[0] for i in ids]); theta=float(s['rotation_angle']); p0=pos[idx]-arr(s['center_world']); sdpos=local(p0,-theta); h=head[idx]+theta
   reconstructed=np.column_stack([sdpos,np.linalg.norm(vel[idx],axis=1),np.cos(h),np.sin(h),shape[idx,:2]])
   errors['position_max_abs_m'].append(float(abs(sdpos-sd[:,:2]).max())); errors['speed_max_abs'].append(float(abs(reconstructed[:,2]-sd[:,2]).max())); errors['heading_max_abs'].append(float(abs(reconstructed[:,3:5]-sd[:,3:5]).max())); errors['size_max_abs_m'].append(float(abs(shape[idx,:2]-sd[:,5:7]).max()))
  n=len(pos); nums.append(n); stats['ego_only_scenes']+=int(n==1); stats['nonfinite_agent_values']+=int((~np.isfinite(np.column_stack([pos,head,vel,shape]))).sum()); stats['nonpositive_length_width']+=int((shape[:,:2]<=0).any(axis=1).sum()); stats['unknown_type']+=int(((typ<0)|(typ>2)).sum())
  xy=local(pos-pos[-1],head[-1]); dh=head-head[-1]; x=np.column_stack([xy,np.cos(dh),np.sin(dh),shape[:,:2],vel]); rows.append(x)
  stats['outside_sd_32m_square']+=int((np.max(abs(xy),axis=1)>32.01).sum())
 x=np.concatenate(rows);allrows[split]=x
 stats.update({'sampled_scenes':len(sample),'sampled_agents':len(x)})
 out[split]={'counts':dict(stats),'agents_per_scene':quant(nums),'map_tokens_per_scene':quant(maps),'reference_time':quant(scene_times),'target_mean':x.mean(0).tolist(),'target_std':x.std(0).tolist(),'position_radius':quant(np.linalg.norm(x[:,:2],axis=1)),'local_lateral_velocity_abs':quant(abs(x[:,7])),'consistency_maxima':{k:max(v) for k,v in errors.items()}}
 print(split,json.dumps(out[split]),flush=True)
ck=torch.load(ROOT/'last.ckpt',map_location='cpu',weights_only=False)
state=ck['state_dict']; prefix='encoder.init_decoder.G1.model.'
mean=arr(state[prefix+'normal_mean']).flatten(); scale=arr(state[prefix+'normal_scale']).flatten(); fresh=allrows['train'].std(0); fresh[:2]*=.5; fresh[2:6]*=2
out['checkpoint']={'epoch':ck['epoch'],'global_step':ck['global_step'],'schedulers':ck['lr_schedulers'],'normal_mean':mean.tolist(),'normal_scale':scale.tolist(),'sample_based_scale_with_existing_heuristic':fresh.tolist(),'scale_ratio_checkpoint_to_sample':(scale/fresh).tolist()}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out,indent=2) + '\n')
print('inventory',json.dumps(out['train_inventory']),flush=True)
print('checkpoint',json.dumps(out['checkpoint']),flush=True)
