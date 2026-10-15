"""Construct the public full-training candidate and exercise native supervision."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']='/tmp/sim_mpl_continuous'
os.environ['OMP_NUM_THREADS']='1'
import gc
gc.disable()
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import datetime,hashlib,importlib.util,json
    import hydra,torch
    from omegaconf import OmegaConf
    import src.smart.model.smart_gail
    import src.smart.datamodules
    from src.smart.diffusion.denoiser import InitDenoiser
    from src.smart.diffusion.initial_diffusion import InitDiffusion

    root=Path(__file__).resolve().parents[1]
    OmegaConf.register_new_resolver('sim_root',lambda:str(root),replace=True)
    torch.set_num_threads(1)
    with hydra.initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
        cfg=hydra.compose(config_name='run.yaml',overrides=[
            'experiment=init_diffusion_657_n24_road_k8_periodic8_full','optimizer=default'])
    original=(InitDenoiser.__init__,InitDiffusion.__init__)
    torch.manual_seed(cfg.seed)
    model=hydra.utils.instantiate(cfg.model,_recursive_=False)
    assert original==(InitDenoiser.__init__,InitDiffusion.__init__)
    decoder=model.encoder.init_decoder
    flow=decoder.G1
    assert sum(p.numel() for p in flow.parameters())==8083978
    assert all(not p.requires_grad for p in model.encoder.map_encoder.parameters())
    assert flow.sampling_integrator=='midpoint' and flow.sampling_time_grid['terminal_time']==.1
    assert decoder.denoiser_options['edge_heading_max_harmonic']==8
    assert cfg.trainer.max_epochs==128 and cfg.trainer.check_val_every_n_epoch==8
    assert cfg.data.num_workers==8 and cfg.data.train_batch_size==512 and cfg.ckpt_path is None
    assert cfg.trainer.limit_val_batches==1.0 and cfg.model.model_config.sd_require_full_set
    spec=importlib.util.spec_from_file_location('ego_fixture',root/'tests/test_init_diffusion_ego_partial.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    x,agent,feature=fixture.InitDiffusionEgoPartialTest.inputs()
    x=torch.cat((x[:,:6],x[:,6:8].norm(dim=-1,keepdim=True)),-1)
    agent['expert_input']=x.clone()
    feature['pt_token']=flow.model.lane_embed(torch.randn(3,128))
    model.train()
    loss=flow._supervised_loss(x,agent,feature)[0].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in flow.parameters())
    assert all(p.grad is None for p in model.encoder.map_encoder.parameters())
    assert flow.model.edge_encoder.r_a2a_emb._periodic_heading_harmonics.max()==8
    assert flow.model.edge_encoder.r_pt2a_emb._periodic_heading_harmonics.max()==8
    base=root/'output/reports/init_diffusion_continuous_20261014'
    OmegaConf.save(cfg,base/'periodic8_full_prepared_config.yaml')
    sources=['tools/periodic_heading_full/model.py','tools/periodic_edge_heading_20261015.py',
             'tools/runtime_periodic_edge_sweep_20261015.py',
             'configs/experiment/init_diffusion_657_n24_road_k8_periodic8_full.yaml']
    result=dict(status='CPU_CONSTRUCTION_AND_BACKWARD_PASS_NOT_DEPLOYED',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),generator_parameters=8083978,
        frozen_map=True,intervals=12,NFE=24,epochs=128,workers=8,check_val_every_n_epoch=8,
        full_validation_scenes=50000,global_constructors_restored=True,finite_native_supervised_gradients=True,
        denoiser_options=decoder.denoiser_options,initial_map_checkpoint_metadata=model.initial_map_checkpoint_metadata,
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sources},
        entrypoint='python src/run.py experiment=init_diffusion_657_n24_road_k8_periodic8_full optimizer=default',
        limitation='Prepared full-training candidate; await later paired evidence and resource review. '
                    'Three-agent CPU backward is a correctness check, not a quality or GPU stability test.')
    (base/'periodic8_full_prepared_config.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['denoiser_options','initial_map_checkpoint_metadata','source_sha256']}))


if __name__=='__main__':main()
