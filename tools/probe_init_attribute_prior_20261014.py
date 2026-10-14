"""CPU-only TRAIN attribute source diagnostics; never fit on validation data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_expanded_prior_mpl')
import gc
gc.disable();gc.collect=lambda *a,**kw:0
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hydra
import torch
from omegaconf import OmegaConf,open_dict
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)


def main():
    out=ROOT/'output/reports/init_diffusion_expanded_20261014/attribute_prior'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Preserve the existing probe')
    manifest=ROOT/'logs/component_loss_scratch/calibration/calibration_files.pkl'
    cfg=OmegaConf.load(ROOT/'logs/init_diffusion_expanded_20261014/roadK8_T4817_2048_N4096_cleanretry1/roadK8/config.yaml')
    with open_dict(cfg):
        cfg.model.model_config.initial_map_checkpoint=None
        cfg.model.model_config.finetune=False
        cfg.data.scenario_dreamer_train_sample_list=str(manifest)
        cfg.data.train_batch_size=64
        cfg.data.num_workers=0
        cfg.data.shuffle=False
    torch.set_num_threads(2);torch.manual_seed(4717)
    model=hydra.utils.instantiate(cfg.model,_recursive_=False)
    dm=hydra.utils.instantiate(cfg.data);dm.setup('fit')
    states=[];labels=[];ego=[];scenes=0
    with torch.no_grad():
        for i,batch in enumerate(dm.train_dataloader()):
            _,agents=model.token_processor(batch)
            model.encoder.init_decoder._prepare_ego_context(agents)
            _,x=model.encoder.init_decoder.G1.model.get_input(agents)
            assert x.device.type=='cpu' and x.shape[-1]==7
            states.append(x[:,4:7].clone());labels.append(agents['type'].clone());ego.append(agents['ego_mask'].clone())
            scenes+=int(agents['num_graphs'])
            if (i+1)%32==0:print('TRAIN scenes',scenes,flush=True)
    values=torch.cat(states);kind=torch.cat(labels);fixed=torch.cat(ego).bool()
    assert scenes==8192 and torch.isfinite(values).all()
    valid=(values[:,:2]>0).all(-1)&(values[:,2]>=0)
    values=values[valid];kind=kind[valid];fixed=fixed[valid]
    torch.save({'attributes':values,'type':kind,'ego_mask':fixed,'fields':['length','width','speed']},out/'train_attributes.pt')
    normal=read(ROOT/'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json')
    mean=torch.tensor(normal['normal_mean'])[0,4:7];std=torch.tensor(normal['normal_scale'])[0,4:7]
    generator=torch.Generator().manual_seed(5917)
    gaussian=torch.randn(values.shape,generator=generator)*std+mean
    qs=torch.tensor([0,.01,.1,.25,.5,.75,.9,.99,1.])
    result=dict(created_utc=now(),source='TRAIN only; exact original8192 calibration membership',
        scenes=scenes,agents=len(values),invalid_agents=int((~valid).sum()),manifest_sha256=sha(manifest),
        data_sha256=sha(out/'train_attributes.pt'),probe_sha256=sha(__file__),
        validation_data_used=False,quantiles=qs.tolist(),groups={})
    for label,name in [(None,'all'),(0,'vehicle'),(1,'pedestrian'),(2,'cyclist')]:
        mask=torch.ones(len(values),dtype=torch.bool) if label is None else kind==label
        v=values[mask]
        result['groups'][name]=dict(count=int(mask.sum()),non_ego_count=int((mask&~fixed).sum()),
            stationary_fraction=float((v[:,2]<.2).float().mean()),quantiles=torch.quantile(v,qs,dim=0).tolist())
    result['gaussian_source']=dict(negative_fraction=(gaussian<0).float().mean(0).tolist(),
        stationary_band_fraction=float((gaussian[:,2].abs()<.2).float().mean()),
        quantiles=torch.quantile(gaussian,qs,dim=0).tolist())
    save(out/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('scenes','agents','gaussian_source')},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
