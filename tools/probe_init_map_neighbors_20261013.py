"""CPU TRAIN-only geometry audit; no training or evaluation weights changed."""
import os
import gc
import faulthandler
from pathlib import Path
import sys
faulthandler.enable();gc.disable();gc.collect=lambda *_,**__:0
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hydra
from omegaconf import OmegaConf,open_dict
import torch
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
from src.smart.utils.map_crop import square_map_mask
from src.smart.utils import transform_to_global, transform_to_local
from tools.run_init_diffusion_657_iteration import read,save,now,sha

ART=ROOT/'output/reports/init_diffusion_evening_20261013/map_neighbor_probe'


def main():
    torch.set_num_threads(1)
    rows=read(ROOT/'output/reports/init_diffusion_evening_20261013/recovery/observations.json')
    row=next(r for r in rows if r['variant']=='lane_only' and r['seed']==4817 and r['budget']==4096)
    cfg_path=Path(row['checkpoint']).parent/'config.yaml';cfg=OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.model.model_config.initial_map_checkpoint=None
        cfg.data.train_batch_size=32;cfg.data.num_workers=0
        cfg.data.pin_memory=False;cfg.data.persistent_workers=False
    torch.manual_seed(10113)
    model=hydra.utils.instantiate(cfg.model,_recursive_=False);model.eval()
    dm=hydra.utils.instantiate(cfg.data,_recursive_=False);dm.setup('fit')
    loader=iter(dm.train_dataloader())
    calibrator=read(ROOT/'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json')
    mu=torch.tensor(calibrator['normal_mean']).reshape(-1)[:2]
    sd=torch.tensor(calibrator['normal_scale']).reshape(-1)[:2]
    totals={str(t):dict(queries=0,road_sources=0,lane_sources=0,old_road_edges=0,new_road_edges=0,new_lane_edges=0,queries_without_road_edges=0) for t in (0.,.25,.5,1.)}
    scenes=0
    with torch.no_grad():
        for batch_i in range(4):
            token_map,agent=model.token_processor(next(loader))
            ego=agent['ego_mask'];ep=agent['initial_pos'][ego];eh=agent['initial_heading'][ego]
            pos=token_map['position'];batch=token_map['batch'];kind=token_map['type'].long()
            crop=square_map_mask(pos,batch,ep,eh,50.)
            roads=crop & ((kind==4)|(kind==5));lanes=crop & ((kind==0)|(kind==1)|(kind==2))
            chosen=roads|lanes
            a_batch=agent['batch'];ap=agent['initial_pos'];ah=agent['initial_heading']
            ap_local,_=transform_to_local(ap,ah,ep[a_batch],eh[a_batch])
            noise=torch.randn(ap_local.shape)*sd+mu
            vehicles=agent['type']==0
            for t in (0.,.25,.5,1.):
                q_local=(1-t)*ap_local+t*noise
                qp,_=transform_to_global(q_local,ah,ep[a_batch],eh[a_batch])
                sums=totals[str(t)]
                for scene in range(len(ep)):
                    q=qp[(a_batch==scene)&vehicles];rp=pos[(batch==scene)&roads];mp=pos[(batch==scene)&chosen]
                    mk=kind[(batch==scene)&chosen]
                    if not len(q):continue
                    sums['queries']+=len(q);sums['road_sources']+=len(rp);sums['lane_sources']+=int(((batch==scene)&lanes).sum())
                    if len(rp):
                        d=torch.cdist(q,rp);v=d.topk(min(30,len(rp)),largest=False).values
                        sums['old_road_edges']+=int(((v<300)&(v>0)).sum())
                    if len(mp):
                        d=torch.cdist(q,mp);v,ix=d.topk(min(30,len(mp)),largest=False)
                        valid=(v<300)&(v>0);r=((mk[ix]==4)|(mk[ix]==5))&valid
                        sums['new_road_edges']+=int(r.sum());sums['new_lane_edges']+=int((valid&~r).sum())
                        sums['queries_without_road_edges']+=int((r.sum(-1)==0).sum())
                    else:sums['queries_without_road_edges']+=len(q)
            scenes+=len(ep)
    for value in totals.values():
        value['road_neighbor_retention']=value['new_road_edges']/max(1,value['old_road_edges'])
        value['lane_neighbor_fraction']=value['new_lane_edges']/max(1,value['new_lane_edges']+value['new_road_edges'])
        value['queries_without_road_fraction']=value['queries_without_road_edges']/max(1,value['queries'])
    save(ART/'results.json',dict(generated_utc=now(),TRAIN_seed=4817,scenes=scenes,config_sha256=sha(cfg_path),time_noise_seed=10113,results=totals,
                               scope='128 TRAIN scenes, CPU, vehicle positions. Geometric query composition only; no attention values, gradients or metric causality claim. Source and evaluation unchanged.'))
    print(totals,flush=True)


if __name__=='__main__':
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
