"""TRAIN-only endpoint action diagnostic; no generation quality inference."""
from pathlib import Path
import os,gc,hashlib,json,pickle,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
gc.disable();gc.collect=lambda *a,**k:0
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 import copy,jinja2,hydra,torch
 from hydra import compose,initialize_config_dir
 from omegaconf import OmegaConf,open_dict
 from torch_geometric.loader import DataLoader
 from src.smart.datasets import MultiDataset
 from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
 from tools.runtime_cached_torch_loader_sweep import install_cached_loader
 install_cached_loader(torch);torch.set_num_threads(1)
 base=ROOT/'output/reports/init_diffusion_continuous_20261014';out=base/'empirical_speed_TRAIN_coupling_v1';out.mkdir(exist_ok=False)
 catalogue=ROOT/'logs/component_loss_scratch/calibration/calibration_files.pkl'
 selected=pickle.loads(catalogue.read_bytes());selected=(selected['files'] if isinstance(selected,dict) else selected)[:128]
 manifest=out/'TRAIN128.pkl';manifest.write_bytes(pickle.dumps(dict(files=selected)))
 OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
 models=[];configs=[]
 for name in ('init_diffusion_omega025_native8','init_diffusion_omega025_empirical_speed_native8'):
  with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):cfg=compose(config_name='run',overrides=['experiment='+name,'optimizer=default'])
  with open_dict(cfg):cfg.paths.output_dir='/tmp/sim_speed_coupling_probe'
  torch.manual_seed(4817);models.append(hydra.utils.instantiate(cfg.model,_recursive_=False).eval());configs.append(cfg)
 dataset=MultiDataset(configs[0].data.train_raw_dir,WaymoTargetBuilderTrain(32),sample_list=str(manifest),sample_list_check_exists=False)
 records=[]
 with torch.no_grad():
  for batch_i,batch in enumerate(DataLoader(dataset,batch_size=32,num_workers=0,shuffle=False)):
   _,agent=models[0].token_processor(batch);models[0].encoder.init_decoder._prepare_ego_context(agent)
   clean,_=models[0].encoder.init_decoder.G1.model.get_input(agent)
   movable=~agent['ego_mask'].bool()
   for draw in range(4):
    samples=[];rngs=[]
    for model in models:
     torch.manual_seed(281700+batch_i*4+draw)
     samples.append(model.encoder.init_decoder.G1._sample_noise(clean,copy.deepcopy(agent)))
     rngs.append(torch.get_rng_state().clone())
    assert torch.equal(rngs[0],rngs[1])
    row=dict(batch=batch_i,draw=draw,agents=int(movable.sum()),arms={})
    for label,noise in zip(('gaussian','empirical_speed'),samples):
     physical=noise[:,:7];scale=models[0].encoder.init_decoder.G1.model.normal_scale
     d=(physical-clean)/scale
     theta=torch.atan2(physical[:,3],physical[:,2])-torch.atan2(clean[:,3],clean[:,2])
     theta=(theta+torch.pi)%(2*torch.pi)-torch.pi
     item=dict(heading_mse=float(theta[movable].square().mean()),heading_over90=float((theta[movable].abs()>torch.pi/2).float().mean()))
     for key,columns in [('position',slice(0,2)),('length',slice(4,5)),('width',slice(5,6)),('speed',slice(6,7))]:
      item[key+'_normalized_mse']=float(d[movable,columns].square().mean())
     row['arms'][label]=item
    records.append(row)
 means={}
 total=sum(r['agents'] for r in records)
 for label in ('gaussian','empirical_speed'):
  means[label]={k:sum(r['agents']*r['arms'][label][k] for r in records)/total for k in records[0]['arms'][label]}
 result=dict(status='TRAIN_ENDPOINT_ACTION_DIAGNOSTIC_COMPLETE_NOT_GENERATION',TRAIN_scenes=128,draws_per_scene=4,agent_draws=total,
  source_endpoint_requires_no_extra_CPU_RNG=True,means=means,ratios_to_gaussian={k:means['empirical_speed'][k]/v for k,v in means['gaussian'].items()},
  limitation='Hungarian coupling changes assignments/bridge targets. Lowerendpointaction doesnotprovebetter learnedJSD; native8 experiment remainsnecessary.',
  validation_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
  source_table_sha256=hashlib.sha256((ROOT/'docs/experiments/init_diffusion_train_attribute_source_full_20261015/quantiles.pt').read_bytes()).hexdigest(),records=records)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True);os._exit(0)

if __name__=='__main__':main()
