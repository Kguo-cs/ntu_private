"""CPU proof of the optional scalar-residual head and its checkpoint guard."""
from pathlib import Path
import copy,hashlib,json,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-floor',type=float,choices=[.05,.2],default=.05)
    options=parser.parse_args()
    import gc;gc.disable()
    import jinja2,torch,hydra
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf,open_dict
    from tools.time_scaled_heading_full.model import install_time_scaled_readout
    from src.smart.diffusion.denoiser import InitDenoiser
    torch.set_num_threads(1);OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        cfg=compose(config_name='run',overrides=['experiment=init_diffusion_657_n24_road_k8_omega025_full','optimizer=default'])
    with open_dict(cfg):cfg.paths.output_dir='/tmp/sim_time_scaled_heading_proof'
    torch.manual_seed(4817);raw=hydra.utils.instantiate(cfg.model,_recursive_=False)
    before={k:v.detach().clone() for k,v in raw.encoder.init_decoder.G1.state_dict().items() if torch.is_tensor(v)}
    rng=torch.get_rng_state().clone();install_time_scaled_readout(raw.encoder.init_decoder,options.time_floor)
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,raw.encoder.init_decoder.G1.state_dict()[k]) for k,v in before.items())
    # Native graph blocks, frame conversion and selected-row beta formatting.
    model=raw.encoder.init_decoder.G1.model
    import importlib.util
    spec=importlib.util.spec_from_file_location('heading_fixture',ROOT/'tests/test_init_diffusion_ego_partial.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    test=fixture.InitDiffusionEgoPartialTest()
    flow=raw.encoder.init_decoder.G1
    state,agent,feature=test.inputs()
    state=torch.cat((state[:,:6],state[:,6:8].norm(dim=-1,keepdim=True)),-1)
    agent['expert_input']=state.clone()
    feature['pt_token']=model.lane_embed(torch.randn(3,128)).detach()
    initial_agent=copy.deepcopy(agent)
    initial_agent['_init_diffusion_type_state']=torch.nn.functional.one_hot(agent['type'],3).float()
    flow.train();losses=flow._supervised_loss(state,agent,feature);loss=losses[0].mean()+losses[1]
    assert torch.isfinite(loss);loss.backward()
    row_grad=model.to_out_m_delta.mlp[-1].weight.grad.abs().sum(-1)
    assert row_grad[3]>0 and row_grad[2]==0
    grads=[p.grad for p in flow.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
    # Call both ordinary and scaled forward on precisely the same latent inputs.
    flow.eval();agent=copy.deepcopy(initial_agent)
    time=torch.tensor([[0.],[.1],[.8]])
    with torch.no_grad():
        plain=InitDenoiser.forward(model,state,time,copy.deepcopy(agent),feature)
        scaled=model(state,time,copy.deepcopy(agent),feature)
    torch.testing.assert_close(scaled[:,:-1],plain[:,:-1],rtol=0,atol=0)
    torch.testing.assert_close(scaled[:,-1:],plain[:,-1:]/time.clamp_min(options.time_floor),rtol=0,atol=0)
    selected=torch.tensor([False,True,True])
    with torch.no_grad():
        masked_plain=InitDenoiser.forward(model,state,time,copy.deepcopy(agent),feature,eval_mask=selected)
        masked=model(state,time,copy.deepcopy(agent),feature,eval_mask=selected)
        copied=copy.deepcopy(model)
        torch.testing.assert_close(copied(state,time,copy.deepcopy(agent),feature),scaled,rtol=0,atol=0)
    torch.testing.assert_close(masked[:,-1:],masked_plain[:,-1:]/time[selected].clamp_min(options.time_floor),rtol=0,atol=0)
    other=copy.deepcopy(raw.encoder.init_decoder)
    extra=copy.deepcopy(other.get_extra_state());extra['denoiser_options'].pop('heading_velocity_parameterization');extra['denoiser_options'].pop('heading_velocity_time_floor')
    try:other.set_extra_state(extra)
    except ValueError:pass
    else:raise AssertionError('A rawomega checkpoint must not silently load into time-scaled inference')
    objective=flow.component_loss.config['components']['heading'];assert objective['kind']=='mse' and objective['time_power']==0 and objective['weight']==.025
    configured=copy.deepcopy(cfg)
    with open_dict(configured):
        configured.model._target_='tools.time_scaled_heading_full.model.TimeScaledCircularVelocitySMART_GAIL'
        configured.model.heading_velocity_time_floor=options.time_floor
    torch.manual_seed(4817)
    fresh=hydra.utils.instantiate(configured.model,_recursive_=False)
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,fresh.encoder.init_decoder.G1.state_dict()[k]) for k,v in before.items())
    assert fresh.encoder.init_decoder.denoiser_options['heading_velocity_parameterization']=='time_scaled'
    wrong_floor=copy.deepcopy(fresh.encoder.init_decoder.get_extra_state())
    wrong_floor['denoiser_options']['heading_velocity_time_floor']=.2 if options.time_floor==.05 else .05
    try:fresh.encoder.init_decoder.set_extra_state(wrong_floor)
    except ValueError:pass
    else:raise AssertionError('A different time floor must not silently load')
    record=dict(status='PASS',parameters=sum(p.numel() for p in flow.parameters()),initial_tensors_and_RNG_unchanged=True,
        native_full_backward_finite=True,physical_and_type_outputs_unchanged=True,scalar_omega_equals_raw_over_clamped_time=True,
        masked_beta_alignment=True,deepcopy_uses_own_model=True,wrong_parameterization_checkpoint_rejected=True,
        objective=objective,time_floor=options.time_floor,wrong_floor_checkpoint_rejected=True,native_full_config_factory=True,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tools/time_scaled_heading_full/model.py']},quality_unproven=True,deployed=False)
    name='time_scaled_heading' if options.time_floor==.05 else 'time_scaled_heading_floor02'
    (ROOT/('output/reports/init_diffusion_continuous_20261014/'+name+'_cpu_proof_20261015.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
