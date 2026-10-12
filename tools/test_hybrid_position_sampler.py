"""CPU-only contracts for the opt-in TEST-only position velocity wrapper."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tools.hybrid_position_sampler_hook import HybridPositionHookCore, hybrid_position_velocity

class HybridTests(unittest.TestCase):
    def setUp(self):
        self.x=torch.arange(28,dtype=torch.float32).reshape(4,7)/10
        self.pred=self.x*.7
        self.time=torch.tensor([[0.],[.025],[.05],[.1]])
        self.v=(self.x-self.pred)/self.time.clamp_min(.05)
    def test_only_position_at_last_lowtime(self):
        out,low=hybrid_position_velocity(self.x,self.time,self.v,self.pred)
        self.assertTrue(torch.equal(low,torch.tensor([False,True,False,False])))
        self.assertTrue(torch.equal(out[:,2:],self.v[:,2:]))
        self.assertTrue(torch.equal(out[[0,2,3]],self.v[[0,2,3]]))
        self.assertTrue(torch.equal(out[1,:2],2*self.v[1,:2]))
    def test_baseline_and_early_return_exact_original_object(self):
        out,_=hybrid_position_velocity(self.x,self.time,self.v,self.pred,enabled=False)
        self.assertIs(out,self.v)
        out,_=hybrid_position_velocity(self.x,torch.full((4,1),.05),self.v,self.pred)
        self.assertIs(out,self.v)
    def test_no_target_mutation_and_final_euler_projection(self):
        pred=self.pred.clone();v=self.v.clone()
        out,_=hybrid_position_velocity(self.x,self.time,self.v,self.pred)
        self.assertTrue(torch.equal(pred,self.pred));self.assertTrue(torch.equal(v,self.v))
        final=self.x[1]-.025*out[1]
        self.assertTrue(torch.allclose(final[:2],self.pred[1,:2]))
        self.assertTrue(torch.equal(final[2:],(self.x[1]-.025*self.v[1])[2:]))
    def test_time_zero_is_unchanged(self):
        out,_=hybrid_position_velocity(self.x,torch.zeros(4,1),self.v,self.pred)
        self.assertIs(out,self.v)
    def test_fit_validation_forbidden_and_invalid_base_refused(self):
        h=HybridPositionHookCore(hybrid_position=True,seed=2817,steps=40,eps=.05,expected_updates=20480,audit_path='/tmp/unused-hybrid.json')
        for method in [h.on_fit_start,h.on_train_start,h.on_validation_start]:
            with self.assertRaises(ValueError):method()
        with self.assertRaises(ValueError):HybridPositionHookCore(seed=2817,steps=20,eps=.05,expected_updates=1,audit_path='/tmp/unused-hybrid.json')
    def test_actual_hook_lifecycle_calls_restores_and_keeps_x0(self):
        class Flow:
            training=False;t_eps=.05;use_sde=False;heading_noise='circular';heading_objective='x0';velocity_representation='speed';size_representation='linear';type_process='separate'
            def _prediction_velocity(f,x,t,p):return (x-p)/t.clamp_min(.05),p
            def sample(f,agent,maps,steps=40):
                x=torch.ones(2,7);p=x*.7
                for i in range(steps):
                    t=torch.full((2,1),1-i/steps);v,out=f._prediction_velocity(x,t,p)
                    assert out is p
                return x
        flow=Flow();orig=flow._prediction_velocity
        module=SimpleNamespace(training=False,encoder=SimpleNamespace(init_decoder=SimpleNamespace(G1=flow,use_ema=True,ema=SimpleNamespace(num_updates=20480),sampling_steps=40,component_loss={'fixed':1})))
        trainer=SimpleNamespace(state=SimpleNamespace(fn='test'),world_size=1,precision='32-true',datamodule=SimpleNamespace(test_batch_size=128))
        with tempfile.TemporaryDirectory() as tmp,patch('tools.evaluate_init_diffusion_sampler.runtime_seed'),patch('tools.evaluate_init_diffusion_sampler.runtime_seed_value',return_value=2817):
            h=HybridPositionHookCore(hybrid_position=True,seed=2817,steps=40,eps=.05,expected_updates=20480,audit_path=Path(tmp)/'audit.json')
            h.on_test_start(trainer,module);flow.sample(None,None,40);h.on_test_batch_end(trainer,module,None,None,0);h.on_test_end(trainer,module)
            self.assertEqual(flow._prediction_velocity,orig);self.assertEqual(flow.t_eps,.05)
            self.assertEqual(h.record['prediction_velocity_calls'],40);self.assertEqual(h.record['hybrid_active_calls'],1)
            self.assertTrue(h.record['prediction_velocity_wrapper_restored']);self.assertTrue(h.record['training_loss_config_unchanged'])
            self.assertEqual(module.encoder.init_decoder.ema.num_updates,20480)

if __name__=='__main__':unittest.main()
