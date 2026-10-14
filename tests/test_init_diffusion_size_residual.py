import unittest
import copy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.smart.diffusion.size_parameterization import size_time_residual_x0
from src.smart.diffusion.denoiser import InitDenoiser
from src.smart.diffusion.initial_diffusion import InitDiffusion


class SizeTimeResidualTest(unittest.TestCase):
    def test_untimed_residual_has_unit_gradient_at_all_times(self):
        raw=torch.randn(3,10,requires_grad=True);latent=torch.randn(3,10)
        time=torch.tensor([[0.],[.3],[1.]])
        model=self.model(shape_x0_parameterization='residual')
        result=model._parameterize_shape_x0(raw,latent,time)
        torch.testing.assert_close(result[:,4:6],latent[:,4:6]+raw[:,4:6])
        torch.testing.assert_close(result[:,:4],raw[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(result[:,6:],raw[:,6:],rtol=0,atol=0)
        result[:,4:6].sum().backward()
        torch.testing.assert_close(raw.grad[:,4:6],torch.ones(3,2),rtol=0,atol=0)
        torch.testing.assert_close(raw.grad[:,:4],torch.zeros(3,4),rtol=0,atol=0)
        torch.testing.assert_close(raw.grad[:,6:],torch.zeros(3,4),rtol=0,atol=0)

    def test_length_only_preserves_width_speed_and_type_fields(self):
        raw=torch.randn(3,10);latent=torch.randn(3,10);time=torch.tensor([[0.],[.3],[1.]])
        model=self.model(shape_x0_parameterization='length_time_residual')
        result=model._parameterize_shape_x0(raw,latent,time)
        torch.testing.assert_close(result[:,4:5],latent[:,4:5]+time*raw[:,4:5])
        torch.testing.assert_close(result[:,:4],raw[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(result[:,5:],raw[:,5:],rtol=0,atol=0)
    def model(self,**options):
        args=dict(token_processor=SimpleNamespace(),input_dim=7,output_dim=7,
                  hidden_dim=32,num_heads=4,num_layers=0,dropout=0.,velocity_representation='speed')
        args.update(options)
        return InitDenoiser(**args)

    def wrapper(self,**options):
        processor=SimpleNamespace(use_refiner=False,learn_init=True,init_map_range=50.)
        args=SimpleNamespace(input_dim=7,hidden_dim=32,num_heads=4,dropout=0.,num_denoiser_layers=1,
                             num_branch_steps=1,branch_steps=[0],sampling_steps=20,use_rl=False,sigma_h=1.)
        with patch.object(InitDiffusion,'_make_args',return_value=args):
            return InitDiffusion(32,4,4,processor,False,velocity_representation='speed',denoiser_num_layers=1,**options)

    def test_default_preserves_seeded_parameters_and_future_rng(self):
        torch.manual_seed(913);a=self.model();after=torch.get_rng_state().clone()
        torch.manual_seed(913);b=self.model(shape_x0_parameterization='identity')
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for name,value in a.state_dict().items():
            if torch.is_tensor(value):torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
        raw=torch.randn(2,7);self.assertIs(a._parameterize_shape_x0(raw,raw,torch.ones(2,1)),raw)

    def test_real_forward_changes_only_size_output_and_adds_no_parameters(self):
        torch.manual_seed(99);a=self.model()
        torch.manual_seed(99);b=self.model(shape_x0_parameterization='time_residual')
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for name,value in a.state_dict().items():
            if torch.is_tensor(value):torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
        state=torch.tensor([[1.,2.,1.,0.,4.,2.,3.],[3.,-2.,0.,1.,5.,2.2,6.]])
        time=torch.tensor([[0.],[.4]])
        agent=dict(batch=torch.zeros(2,dtype=torch.long),type=torch.zeros(2,dtype=torch.long),num_graphs=1,
                   ego_mask=torch.tensor([True,False]),ego_feat=torch.tensor([[0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,0.,0.]]))
        raw=torch.ones(2,7)
        outputs=[]
        for model in (a,b):
            with patch.object(model,'_apply_graph_attention',return_value=raw.clone()):
                outputs.append(model(state,time,copy.deepcopy(agent),{},use_map_condition=False))
        torch.testing.assert_close(outputs[1][:,4:6],state[:,4:6]+time*raw[:,4:6])
        torch.testing.assert_close(outputs[0][:,:4],outputs[1][:,:4],rtol=0,atol=0)
        torch.testing.assert_close(outputs[0][:,6:],outputs[1][:,6:],rtol=0,atol=0)

    def test_decoded_x0_keeps_original_reconstruction_objective(self):
        from src.smart.diffusion.diffusion_utils import get_diff_loss
        clean=torch.tensor([[1.,2.,1.,0.,4.,2.,3.],[3.,-2.,0.,1.,5.,2.2,6.]])
        time=torch.tensor([[.3],[.7]]);latent=clean+.2;desired=clean+.4
        raw=desired.clone();raw[:,4:6]=(desired[:,4:6]-latent[:,4:6])/time
        decoded=self.model(shape_x0_parameterization='time_residual')._parameterize_shape_x0(raw,latent,time)
        agent=dict(batch=torch.zeros(2,dtype=torch.long))
        expected=get_diff_loss(agent,desired,clean,time,.05,x_pred=True,physical_state_dim=7)
        actual=get_diff_loss(agent,decoded,clean,time,.05,x_pred=True,physical_state_dim=7)
        for first,second in zip(expected,actual):torch.testing.assert_close(first,second)

    def test_legacy_checkpoint_loads_default_but_rejects_changed_coordinates(self):
        state=copy.deepcopy(self.wrapper().state_dict())
        state['_extra_state']['denoiser_options'].pop('shape_x0_parameterization')
        self.wrapper().load_state_dict(state,strict=True)
        with self.assertRaisesRegex(ValueError,'denoiser options'):
            self.wrapper(shape_x0_parameterization='time_residual').load_state_dict(state,strict=True)

    def test_invalid_mode_or_nondeterministic_layout_is_rejected(self):
        for value in ('bad',None,True):
            with self.assertRaises(ValueError):self.model(shape_x0_parameterization=value)
        with self.assertRaises(ValueError):
            self.model(shape_x0_parameterization='time_residual',velocity_representation='vector',
                       input_dim=8,output_dim=8,x_pred=False)
    def test_endpoints_preserve_non_size_and_type_features(self):
        latent=torch.tensor([[2.,3.,1.,0.,4.,2.,6.,.1,.2,.7],
                             [1.,0.,0.,1.,5.,2.2,0.,.8,.1,.1]])
        raw=torch.ones_like(latent)
        time=torch.tensor([[0.],[1.]])
        output=size_time_residual_x0(raw,latent,time)
        torch.testing.assert_close(output[:,4:6],latent[:,4:6]+time,rtol=0,atol=0)
        torch.testing.assert_close(output[:,:4],raw[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(output[:,6:],raw[:,6:],rtol=0,atol=0)

    def test_size_gradient_scales_with_time_without_other_field_changes(self):
        raw=torch.randn(3,8,requires_grad=True)
        latent=torch.randn(3,8)
        time=torch.tensor([[0.],[.3],[1.]])
        output=size_time_residual_x0(raw,latent,time)
        output[:,4:6].sum().backward()
        torch.testing.assert_close(raw.grad[:,4:6],time.expand(-1,2),rtol=0,atol=0)
        torch.testing.assert_close(raw.grad[:,:4],torch.zeros(3,4),rtol=0,atol=0)
        torch.testing.assert_close(raw.grad[:,6:],torch.zeros(3,2),rtol=0,atol=0)

    def test_log_sizes_stay_encoded_until_existing_physical_decode(self):
        latent=torch.tensor([[0.,0.,1.,0.,4.,1.8,3.]])
        latent[:,4:6]=latent[:,4:6].log()
        target=torch.tensor([[4.8,2.1]])
        time=torch.tensor([[.4]])
        raw=torch.zeros_like(latent)
        raw[:,4:6]=(target.log()-latent[:,4:6])/time
        output=size_time_residual_x0(raw,latent,time)
        torch.testing.assert_close(output[:,4:6].exp(),target)
        self.assertTrue((output[:,4:6].exp()>0).all())


if __name__=='__main__':unittest.main()
