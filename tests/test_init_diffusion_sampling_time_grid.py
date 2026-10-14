import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.smart.diffusion.sampling_time_grid import build_sampling_time_grid,parse_sampling_time_grid


class SamplingTimeGridTest(unittest.TestCase):
    def grid(self,config=None,steps=20):
        return build_sampling_time_grid(steps,device='cpu',dtype=torch.float32,config=config)

    def test_default_and_explicit_uniform_are_bitwise_legacy_and_leave_rng_unchanged(self):
        before=torch.get_rng_state().clone();expected=torch.linspace(1.,0.,21)
        for c in (None,{},dict(kind='power',power=1)):
            self.assertTrue(torch.equal(self.grid(c),expected))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_nonuniform_grids_keep_twenty_intervals_and_exact_endpoints(self):
        for c in [dict(power=.75),dict(power=1.5),dict(kind='cosine')]:
            t=self.grid(c)
            self.assertEqual(len(t)-1,20)
            self.assertEqual(float(t[0]),1.)
            self.assertEqual(float(t[-1]),0.)
            self.assertTrue((t[:-1]>t[1:]).all())

    def test_terminal_control_keeps_final_denoising_noise_comparable(self):
        for c in [dict(power=.75),dict(power=1.5),dict(kind='cosine')]:
            t=self.grid(dict(c,terminal_time=.05))
            self.assertAlmostEqual(float(t[-2]),.05,places=7)
            self.assertEqual(float(t[0]),1.)
            self.assertEqual(float(t[-1]),0.)
            self.assertTrue((t[:-1]>t[1:]).all())

    def test_malformed_or_precision_collapsed_grids_fail(self):
        for c in [{'kind':'unknown'},{'power':0},{'power':True},{'power':float('nan')},
                  {'terminal_time':0},{'terminal_time':1},{'extra':1},
                  {'kind':'cosine','power':2},{'power':100}]:
            with self.subTest(c=c),self.assertRaises(ValueError):self.grid(c)
        with self.assertRaises(ValueError):self.grid({'terminal_time':.05},steps=1)

    def flow(self,config=None):
        from src.smart.diffusion.scale_flow import Flow
        args=SimpleNamespace(input_dim=8,hidden_dim=32,num_heads=2,dropout=0.,
                             num_denoiser_layers=0,num_branch_steps=1,branch_steps=[0],
                             sampling_steps=20,use_rl=False,sigma_h=1.,sampling_time_grid=config)
        processor=SimpleNamespace(use_refiner=False,learn_init=True,init_map_range=50.)
        return Flow(args,processor,False)

    def test_inference_grid_adds_no_parameters_and_preserves_seeded_weights(self):
        torch.manual_seed(817);a=self.flow();after=torch.get_rng_state().clone()
        torch.manual_seed(817);b=self.flow({'power':.75})
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for key,value in a.state_dict().items():
            if torch.is_tensor(value):self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))

    def test_actual_flow_sampler_uses_requested_twenty_or_twentyfour_model_calls(self):
        state=torch.tensor([[2.,3.,1.,0.,4.5,2.,5.,0.],
                            [0.,0.,1.,0.,4.5,2.,5.,0.],
                            [-4.,7.,0.,1.,4.,1.8,3.,0.]])
        for steps in (20,24):
            for config in [None,{'power':.75},{'kind':'cosine','terminal_time':.05}]:
                flow=self.flow(config);flow.eval()
                agent=dict(batch=torch.zeros(3,dtype=torch.long),type=torch.tensor([0,1,2]),
                           ego_mask=torch.tensor([False,True,False]),num_graphs=1,
                           expert_input=state.clone())
                with patch.object(flow.model,'forward',return_value=state.clone()) as model:
                    result=flow.sample(agent,{},steps=steps)
                self.assertEqual(model.call_count,steps)
                # Legacy t_eps=.05 intentionally leaves residual when the
                # last grid point is smaller than .05 (uniform N24). Check
                # that behavior explicitly rather than hiding it in tolerance.
                times=self.grid(config,steps=steps)
                retained=(1-(times[:-1]-times[1:])/times[:-1].clamp_min(.05)).prod()
                initial=model.call_args_list[0].args[0]
                expected=state+retained*(initial-state)
                torch.testing.assert_close(result,expected,rtol=0,atol=2e-6)

    def test_24_step_terminal_control_avoids_legacy_floor_residual(self):
        for steps,config in [(20,None),(24,None),(24,{'terminal_time':.05})]:
            times=self.grid(config,steps=steps)
            retained=(1-(times[:-1]-times[1:])/times[:-1].clamp_min(.05)).prod()
            if steps==24 and config is None:
                self.assertGreater(float(retained),0.)
            else:
                self.assertEqual(float(retained),0.)


if __name__=='__main__':unittest.main()
