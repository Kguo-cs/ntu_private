"""Check independent TRAIN seed with unchanged recipe and frozen AIRL map."""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
gc.disable()


def main():
    import jinja2,hydra,torch
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf,open_dict
    from tools.run_native_prefix_training_20261015 import digest_parameters
    torch.set_num_threads(1)
    OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
    cfgs=[];models=[]
    for name in ('init_diffusion_657_n24_road_k8_omega025_full','init_diffusion_657_n24_road_k8_omega025_seed6817_full'):
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
            cfg=compose(config_name='run',overrides=['experiment='+name,'optimizer=default'])
        with open_dict(cfg):cfg.paths.output_dir='/tmp/sim_omega025_replica_cpu'
        torch.manual_seed(cfg.seed)
        cfgs.append(OmegaConf.to_container(cfg,resolve=True))
        models.append(hydra.utils.instantiate(cfg.model,_recursive_=False))
    diffs=[k for k in cfgs[0] if cfgs[0][k]!=cfgs[1][k]]
    assert diffs==['task_name','seed','callbacks','logger'],diffs
    # Resolved callback filenames andW&B names inherit the new task name.
    # Require these to be metadata-only differences, not a changed recipe.
    def normalize(value):
        if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
        if isinstance(value,list):return [normalize(v) for v in value]
        if isinstance(value,str):return value.replace(cfgs[1]['task_name'],cfgs[0]['task_name'])
        return value
    normalized=normalize(cfgs[1]);normalized['seed']=cfgs[0]['seed']
    assert normalized==cfgs[0], 'Numerical config changed beyond independent seed'
    first,model=models
    assert digest_parameters(first.encoder.map_encoder)==digest_parameters(model.encoder.map_encoder)
    assert digest_parameters(first.encoder.init_decoder.G1)!=digest_parameters(model.encoder.init_decoder.G1)
    assert all(not p.requires_grad for p in model.encoder.map_encoder.parameters())
    flow=model.encoder.init_decoder.G1
    assert sum(p.numel() for p in flow.parameters())==8083978
    assert flow.model.attribute_source is None
    assert model.encoder.init_decoder.ema.num_updates==0
    assert cfgs[1]['trainer']['max_epochs']==model.lr_total_steps==128
    spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test_init_diffusion_ego_partial.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    state,agent,features=fixture.InitDiffusionEgoPartialTest().inputs()
    state=torch.cat((state[:,:6],state[:,6:8].norm(dim=-1,keepdim=True)),-1);agent['expert_input']=state.clone()
    torch.manual_seed(142015);features['pt_token']=flow.model.lane_embed(torch.randn(3,128)).detach()
    result=flow._supervised_loss(state,agent,features);loss=result[0].mean()+result[1]
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None)
    paths=[ROOT/'configs/experiment/init_diffusion_657_n24_road_k8_omega025_seed6817_full.yaml',Path(__file__),
        ROOT/'tools/shared_heading_full/model.py',ROOT/'tools/shared_heading_velocity_20261015.py']
    paths+=sorted((ROOT/'src/smart').rglob('*.py'))
    record=dict(status='OMEGA025_INDEPENDENT_TRAIN_SEED_CPU_PREFLIGHT_PASS',seed=6817,
        config_only_differences=diffs,parameters=8083978,initial_G1_sha256=digest_parameters(flow),
        initial_map_sha256=digest_parameters(model.encoder.map_encoder),initial_G1_differs=True,
        frozen_map_exact=True,EMA_updates=0,source='gaussian',native_backward_finite=True,
        training_epochs=128,LR_horizon_epochs=128,predeclared_checkpoint_epochs=[96,104],GEN=[2817,3817],
        evaluation='Complete same50000 scenes andcurrentagentmetrics, nativeMP24, batch128, medium32true, fixedegoPSHtype',
        quality_unproven=True,no_automatic_failure_retries=True,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out=ROOT/'output/reports/init_diffusion_continuous_20261014/omega025_seed_replication_cpu_proof_20261015.json'
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='source_hashes'}),flush=True);os._exit(0)


if __name__=='__main__':main()
