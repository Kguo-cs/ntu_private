"""CPU-only callback, checkpoint metadata and native-report certification tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools import evaluate_init_diffusion_n20_power_grids as driver
from tools.n20_power_grid_sampler import REVIEWED_RECIPE,power_time_grid


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(3))
        self.register_buffer('normal_mean',torch.ones(1,7));self.register_buffer('normal_scale',torch.ones(1,7))
        self.m_delta_dim=7;self.eval()
    def forward(self,latent,time,*args):return latent


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=Model();self.t_eps=.05;self.use_sde=False;self.use_refiner=False
        for name,value in REVIEWED_RECIPE.items():setattr(self,name,value)
        self.eval()
    def sample(self,agent,feature,steps=20,branch_steps=None):return 'original class binding'


def cell(scenes=2,batch=2,arm=None):
    return {**(arm or driver.ARMS[0]),'TRAIN_seed':4817,'GEN_seed':2817,'trained_updates':4096,
        'sampling_steps':20,'generator_parameter_count':3,'scenes':scenes,'evaluation_batch_size':batch,
        'native_metric_identity':{'metric_scope':'fixture','map_source':'reference','reference_mode':'matched'}}


def native_fixture_sampler(flow,agent,feature,*,power,steps,denominator_policy,audit):
    grid=power_time_grid(power,device='cpu',dtype=torch.float32)
    state=torch.ones(len(agent['batch']),7)
    for time in grid[:-1]:flow.model(state,torch.full((len(state),1),time))
    audit.update(actual_denoiser_forward_calls=20,all_output_values_finite=True,power=power,
                 denominator_policy=denominator_policy,effective_grid=grid.tolist())
    return state


def environment():
    flow=Flow();decoder=NS(G1=flow,use_ema=True,ema=NS(num_updates=4096),sampling_steps=20,component_loss={'fixture':1})
    model=NS(training=False,strict_loading=True,encoder=NS(init_decoder=decoder))
    trainer=NS(state=NS(fn='test'),world_size=1,precision='32-true',datamodule=NS(test_batch_size=2))
    return flow,decoder,model,trainer


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'audit.json'
        self.seed=patch.object(driver,'runtime_seed');self.seed_value=patch.object(driver,'runtime_seed_value',return_value=2817)
        self.sampler=patch.object(driver,'sample_with_power_grid',side_effect=native_fixture_sampler)
        for context in (self.seed,self.seed_value,self.sampler):context.start();self.addCleanup(context.stop)

    def start(self,arm=None,scenes=2):
        flow,decoder,model,trainer=environment()
        hook=driver.PowerGridHookCore(cell=cell(scenes=scenes,arm=arm),audit_path=self.path)
        hook.on_test_start(trainer,model)
        return hook,flow,decoder,model,trainer

    def one_batch(self,hook,flow,model,trainer,index,scenes):
        batch=NS(num_graphs=scenes);hook.on_test_batch_start(trainer,model,batch,index)
        flow.sample({'batch':torch.zeros(scenes,dtype=torch.long)},{},20,None)
        hook.on_test_batch_end(trainer,model,None,batch,index)

    def test_actual_twenty_forwards_and_exact_class_binding_restoration(self):
        hook,flow,decoder,model,trainer=self.start()
        self.assertIn('sample',flow.__dict__)
        self.one_batch(hook,flow,model,trainer,0,2);hook.on_test_end(trainer,model)
        self.assertNotIn('sample',flow.__dict__);self.assertEqual(flow.sample(None,None),'original class binding')
        report=json.loads(self.path.read_text())
        self.assertTrue(report['finished']);self.assertTrue(report['sample_binding_restored'])
        self.assertEqual(report['denoiser_forward_calls'],20);self.assertEqual(report['generated_scene_count'],2)
        self.assertEqual(len(report['per_batch']),1);self.assertNotIn(hook,driver.ACTIVE_HOOKS)

    def test_all_four_arm_policies_pass_explicitly_to_sampler(self):
        for arm in driver.ARMS:
            with self.subTest(arm=arm['arm']):
                hook,flow,decoder,model,trainer=self.start(arm)
                self.one_batch(hook,flow,model,trainer,0,2);hook.on_test_end(trainer,model)
                actual=hook.record['per_batch'][0]['sampler']
                self.assertEqual(actual['power'],arm['power']);self.assertEqual(actual['denominator_policy'],arm['denominator_policy'])
                self.assertEqual(flow.t_eps,.05)

    def test_existing_instance_binding_and_hook_preserved_on_exception(self):
        flow,decoder,model,trainer=environment();original=lambda *a,**k:'existing instance'
        flow.sample=original;handle=flow.model.register_forward_pre_hook(lambda *a:None)
        hooks_before=dict(flow.model._forward_pre_hooks)
        hook=driver.PowerGridHookCore(cell=cell(),audit_path=self.path)
        hook.on_test_start(trainer,model)
        batch=NS(num_graphs=2);hook.on_test_batch_start(trainer,model,batch,0)
        def broken(*args,**kwargs):
            flow.model(torch.ones(2,7),torch.ones(2,1));raise RuntimeError('sampler failure')
        with patch.object(driver,'sample_with_power_grid',side_effect=broken):
            with self.assertRaisesRegex(RuntimeError,'sampler failure'):flow.sample({'batch':torch.zeros(2,dtype=torch.long)},{})
        hook.on_exception(trainer,model,RuntimeError('sampler failure'))
        self.assertIs(flow.sample,original);self.assertEqual(dict(flow.model._forward_pre_hooks),hooks_before)
        self.assertFalse(json.loads(self.path.read_text())['finished']);self.assertEqual(flow.t_eps,.05);handle.remove()

    def test_nfe_mismatch_is_rejected_and_global_cleanup_restores(self):
        hook,flow,decoder,model,trainer=self.start();batch=NS(num_graphs=2)
        hook.on_test_batch_start(trainer,model,batch,0)
        flow.sample({'batch':torch.zeros(2,dtype=torch.long)},{})
        flow.model(torch.ones(2,7),torch.ones(2,1))
        with self.assertRaisesRegex(ValueError,'NFE'):hook.on_test_batch_end(trainer,model,None,batch,0)
        driver.restore_active_hooks();self.assertNotIn('sample',flow.__dict__)

    def test_incomplete_test_never_finished_and_restores(self):
        hook,flow,decoder,model,trainer=self.start(scenes=4)
        self.one_batch(hook,flow,model,trainer,0,2)
        with self.assertRaisesRegex(ValueError,'incomplete'):hook.on_test_end(trainer,model)
        self.assertFalse(hook.record['finished']);self.assertNotIn('sample',flow.__dict__)

    def test_tail_batch_counts_actual_scenes(self):
        hook,flow,decoder,model,trainer=self.start(scenes=3)
        self.one_batch(hook,flow,model,trainer,0,2);self.one_batch(hook,flow,model,trainer,1,1)
        hook.on_test_end(trainer,model)
        self.assertEqual(hook.record['generated_scene_count'],3);self.assertEqual(hook.record['denoiser_forward_calls'],40)

    def test_weight_or_objective_changes_rejected_with_restoration(self):
        for change in ('weight','objective','ema'):
            with self.subTest(change=change):
                hook,flow,decoder,model,trainer=self.start();self.one_batch(hook,flow,model,trainer,0,2)
                if change=='weight':
                    with torch.no_grad():flow.model.weight.add_(1)
                elif change=='objective':decoder.component_loss={'changed':2}
                else:decoder.ema.num_updates=4097
                with self.assertRaises(ValueError):hook.on_test_end(trainer,model)
                self.assertNotIn('sample',flow.__dict__);self.assertFalse(hook.record['finished'])

    def test_inference_only_and_bad_runtime_modes_rejected(self):
        hook=driver.PowerGridHookCore(cell=cell(),audit_path=self.path)
        for method in (hook.on_fit_start,hook.on_train_start,hook.on_validation_start):
            with self.assertRaises(ValueError):method()
        for change in ('precision','ema','batch','strict'):
            flow,decoder,model,trainer=environment()
            if change=='precision':trainer.precision='16-mixed'
            elif change=='ema':decoder.ema.num_updates=2048
            elif change=='batch':trainer.datamodule.test_batch_size=128
            else:model.strict_loading=False
            with self.assertRaises(ValueError):hook.on_test_start(trainer,model)
            self.assertNotIn('sample',flow.__dict__)


class CheckpointMetadataTests(unittest.TestCase):
    def make_payload(self):
        calibration=driver.read(driver.CALIBRATION)
        state={f'encoder.init_decoder.G1.model.{name}':torch.tensor(calibration[name],dtype=torch.float32)
               for name in ('normal_mean','normal_scale')}
        state['encoder.init_decoder._extra_state']={'ema':{'num_updates':4096}}
        return {'global_step':4096,'state_dict':state,'scratch_sweep':{'seed':4817,'saved_global_step':4096,
                                                                    'actual_train_offset':0,'resumed_from':None}}
    def test_actual_step_ema_seed_and_normalizer_without_refitting(self):
        payload=self.make_payload();before={k:v.clone() for k,v in payload['state_dict'].items() if torch.is_tensor(v)}
        result=driver.validate_checkpoint_payload(payload,cell())
        self.assertTrue(result['normalizer_verified_without_recalibration'])
        for k,v in before.items():self.assertTrue(torch.equal(v,payload['state_dict'][k]))
    def test_mutated_endpoint_ema_seed_and_statistics_rejected(self):
        for change in ('step','ema','seed','normalizer'):
            payload=self.make_payload()
            if change=='step':payload['global_step']=2048
            elif change=='ema':payload['state_dict']['encoder.init_decoder._extra_state']['ema']['num_updates']=2048
            elif change=='seed':payload['scratch_sweep']['seed']=5817
            else:payload['state_dict']['encoder.init_decoder.G1.model.normal_mean'][0,0]+=1
            with self.assertRaises(ValueError):driver.validate_checkpoint_payload(payload,cell())


if __name__=='__main__':unittest.main()
