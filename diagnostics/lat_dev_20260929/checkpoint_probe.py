import sys, json, random, math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
sys.path.insert(0,'/home/ke/code/sim')
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData, Batch
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.metrics.sd_lane_helpers import resample_lanes
root=Path('/home/ke/code/sim'); out=Path('/tmp/sim_lat_audit')
torch.set_num_threads(2)
ck=torch.load(root/'src/waymo_data/last.ckpt',map_location='cpu',weights_only=False)
cfg=OmegaConf.create(ck['hyper_parameters']['model_config'])
tp=TokenProcessor(**cfg.token_processor).cuda().eval()
model=SMARTDecoder(**cfg.decoder,token_processor=tp,n_token_agent=tp.n_token_agent,finetune=cfg.finetune).cuda().eval()
keys={k[len('encoder.'):]:v for k,v in ck['state_dict'].items() if k.startswith('encoder.')}
incompatible=model.load_state_dict(keys,strict=False)
print('checkpoint',ck['epoch'],ck['global_step'],'missing',incompatible.missing_keys,'unexpected',incompatible.unexpected_keys,flush=True)
assert not incompatible.missing_keys
init=model.init_decoder;flow=init.G1
assert not flow.use_sde and not flow.use_refiner
paths=random.Random(817).sample(sorted((root/'src/waymo_data/full/scenario_dreamer_val').glob('*.pt')),128)
results={name:[] for name in ['gt','steps10','steps20','steps50','steps10_last005','denoise_t005','denoise_t01']}
nonresults={name:[] for name in results}; errors={};plot_records=[]
with torch.inference_mode():
 for start in range(0,len(paths),32):
  docs=[torch.load(p,map_location='cpu',weights_only=False) for p in paths[start:start+32]]
  graphs=[HeteroData({k:d[k] for k in ['agent','map_save','pt_token','scene_timestep']}) for d in docs]
  batch=Batch.from_data_list(graphs).cuda()
  tm,ta=tp(batch)
  ta['map_feature']=model.map_encoder(tm)
  pos,head,ab,ng=init._prepare_ego_context(ta)
  mf=init._initial_map_feature(ta,pos,head,ng)
  clean,_=flow.model.get_input(ta);ta['expert_input']=clean
  generated={'gt':clean.cpu().numpy()}
  for steps in [10,20,50]:
   torch.manual_seed(817+start)
   generated[f'steps{steps}']=flow.sample(ta,mf,steps=steps).cpu().numpy()
  torch.manual_seed(817+start)
  latent=flow.model.denormalize(torch.randn_like(clean))
  times=torch.cat([torch.linspace(1,.2,9,device='cuda'),torch.tensor([.05,0.],device='cuda')])
  for i in range(len(times)-1):latent,*_=flow._sample_step(latent,times[i],times[i+1],ta,mf)
  latent[ta['ego_mask']]=clean[ta['ego_mask']]
  generated['steps10_last005']=latent.cpu().numpy()
  for t in [.05,.1]:
   torch.manual_seed(817+start)
   noise=flow.model.denormalize(torch.randn_like(clean))
   ts=torch.full((len(clean),1),t,device='cuda');latent=(1-ts)*clean+ts*noise
   latent[ta['ego_mask']]=clean[ta['ego_mask']];ts[ta['ego_mask']]=0
   _,pred=flow._model_velocity(latent,ts,ta,mf);pred[ta['ego_mask']]=clean[ta['ego_mask']]
   name='denoise_t005' if t==.05 else 'denoise_t01';generated[name]=pred.cpu().numpy()
   mask=(ta['type']==0)&~ta['ego_mask'];errors.setdefault(name,[]).extend(torch.linalg.vector_norm(pred[mask,:2]-clean[mask,:2],dim=-1).cpu().tolist())
  for i,d in enumerate(docs):
   lanes=resample_lanes(np.asarray(d['scenario_dreamer']['road_points']),num_points=100)
   lanes=np.stack([lanes[...,1],-lanes[...,0]],axis=-1).reshape(-1,2)
   vehicle=((ta['batch']==i)&(ta['type']==0)).cpu().numpy()
   nonvehicle=vehicle&(~ta['ego_mask']).cpu().numpy()
   for name,g in generated.items():
    dist=np.linalg.norm(g[vehicle,None,:2]-lanes[None,:,:],axis=-1).min(1);results[name].extend(dist.tolist())
    distn=np.linalg.norm(g[nonvehicle,None,:2]-lanes[None,:,:],axis=-1).min(1) if nonvehicle.any() else []
    nonresults[name].extend(np.asarray(distn).tolist())
  print('done',start+len(docs),flush=True)
np.savez(out/'checkpoint_distances.npz',**{k:np.array(v) for k,v in results.items()},**{k+'_non_ego':np.array(v) for k,v in nonresults.items()})
bins=np.arange(0,1.5001,.1)
def stats(values,gt):
 a=np.asarray(values);g=np.asarray(gt);ar=a[a<=1.5];gr=g[g<=1.5]
 h=np.histogram(ar,bins)[0];gh=np.histogram(gr,bins)[0]
 return {'vehicles':len(a),'onroad_fraction':float(np.mean(a<=1.5)),'distance_mean_all':float(a.mean()),'distance_median_all':float(np.median(a)),'distance_mean_onroad':float(ar.mean()),'histogram':h.tolist(),'lat_dev_jsd':float(jensenshannon(h,gh)**2*10)}
summary={'checkpoint':'src/waymo_data/last.ckpt','epoch':ck['epoch'],'global_step':ck['global_step'],'sample_count':len(paths),'sample_files':[p.name for p in paths],'all_vehicles':{k:stats(v,results['gt']) for k,v in results.items()},'non_ego_vehicles':{k:stats(v,nonresults['gt']) for k,v in nonresults.items()},'denoise_xy_errors':{k:{'mean':float(np.mean(v)),'median':float(np.median(v)),'p95':float(np.quantile(v,.95))} for k,v in errors.items()}}
(out/'checkpoint_probe.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='sample_files'},indent=2))
