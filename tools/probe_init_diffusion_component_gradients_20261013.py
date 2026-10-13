"""CPU TRAIN-only diagnostic, not a generated-quality experiment.

Reweights the same prediction graph to measure output-MLP gradient scale and
direction. It does not update any weights or alter saved training/evaluation.
"""
import copy
import os
from pathlib import Path
import sys
import gc
import faulthandler
faulthandler.enable()
gc.disable()
gc.collect=lambda *args,**kwargs:0
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/sim_evening_mpl'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hydra
import lightning as L
from omegaconf import OmegaConf,open_dict
import torch
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
from src.smart.diffusion.component_loss import ComponentLoss
from tools.run_init_diffusion_657_iteration import read,save,sha,now

ART=ROOT/'output/reports/init_diffusion_evening_20261013/component_gradient_probe'

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('medium')
    rows=read(ROOT/'output/reports/init_diffusion_jsd_round2_20261013/speed_isolation/observations.json')
    row=next(r for r in rows if r['variant']=='lane_only' and r['seed']==5817 and r['budget']==4096)
    cfg=OmegaConf.load(Path(row['checkpoint']).parent/'config.yaml')
    with open_dict(cfg):
        cfg.model.model_config.initial_map_checkpoint=None
        cfg.model.model_config.decoder.init_diffusion.normalizer_path=None
        cfg.data.train_batch_size=32;cfg.data.num_workers=0
        cfg.data.pin_memory=False;cfg.data.persistent_workers=False
    L.seed_everything(5817,workers=True)
    model=hydra.utils.instantiate(cfg.model,_recursive_=False)
    state=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict'],strict=True);del state
    model.train();model.encoder.map_encoder.eval()
    for p in model.encoder.map_encoder.parameters():p.requires_grad_(False)
    model._log_train=lambda *args,**kwargs:None
    dm=hydra.utils.instantiate(cfg.data,_recursive_=False);dm.setup('fit')
    loader=iter(dm.train_dataloader())
    flow=model.encoder.init_decoder.G1;objective=flow.component_loss
    original=objective.regression
    variants={}
    for name,component,power in [('speed_p0','velocity',0),('heading_p1','heading',1),('length_p0','length',0)]:
        c=copy.deepcopy(objective.config);c['components'][component]['time_power']=power
        variants[component]=(name,ComponentLoss(c))
    captured={}
    def observe(name,*args,**kwargs):
        raw,weighted=original(name,*args,**kwargs);captured[name]=(raw,weighted)
        if name in variants:
            label,other=variants[name];captured[label]=other.regression(name,*args,**kwargs)
        return raw,weighted
    objective.regression=observe
    parameter=flow.model.to_out_m_delta.mlp[0].weight
    results=[]
    for i in range(4):
        captured.clear();model.zero_grad(set_to_none=True)
        torch.manual_seed(10113+i)
        loss=model.training_step(next(loader),i)
        gradients={};stats={}
        for name,(raw,weighted) in captured.items():
            grad=torch.autograd.grad(weighted,parameter,retain_graph=True,allow_unused=True)[0]
            if grad is None:continue
            vec=grad.detach().flatten();gradients[name]=vec
            stats[name]=dict(raw_loss=float(raw.detach()),weighted_loss=float(weighted.detach()),output_hidden_gradient_norm=float(vec.norm()))
        cosine=lambda a,b:float(torch.dot(gradients[a],gradients[b])/(gradients[a].norm()*gradients[b].norm()).clamp_min(1e-12))
        conflicts={f'{a}_vs_{b}':cosine(a,b) for a,b in [('velocity','heading'),('velocity','length'),('speed_p0','heading'),('speed_p0','length'),('heading','length')]}
        result=dict(batch=i,TRAIN_seed=5817,time_noise_seed=10113+i,batch_size=32,stats=stats,output_hidden_cosines=conflicts)
        results.append(result);print(result,flush=True)
        save(ART/'results.json',dict(generated_utc=now(),checkpoint=row['checkpoint'],checkpoint_sha256=sha(row['checkpoint']),scope='Four32-scene TRAIN minibatches; CPU; one shared output-MLP parameter; no updates; not GPU JSD ranking evidence',results=results))
        del loss
    objective.regression=original

if __name__=='__main__':main()
