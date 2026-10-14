import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.smart.diffusion.attribute_source import AttributeQuantileSource


class AttributeSourceTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.path=Path(self.directory.name)/'train.pt'
        self.values=torch.tensor([[1.,.5,0.],[3.,1.,0.],[5.,2.,2.],[7.,3.,10.]])
        torch.save(dict(attributes=self.values,fields=['length','width','speed']),self.path)
        self.options=dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),knots=13)

    def tearDown(self):self.directory.cleanup()

    def test_enabled_fields_only_and_no_extra_rng_or_parameters(self):
        source=AttributeQuantileSource(self.options)
        standard=torch.randn(100,7);physical=torch.randn(100,7)
        before=torch.get_rng_state().clone();actual=source(standard,physical)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(actual[:,:6],physical[:,:6]))
        self.assertTrue(torch.equal(actual,source(standard,physical)))
        self.assertGreaterEqual(float(actual[:,6].min()),0.)
        self.assertLessEqual(float(actual[:,6].max()),10.)
        self.assertEqual(sum(p.numel() for p in source.parameters()),0)

    def test_cdf_extremes_and_median_preserve_physical_units(self):
        source=AttributeQuantileSource(dict(self.options,shape=True))
        z=torch.zeros(3,7);z[0,4:]=-20;z[2,4:]=20
        out=source(z,z)
        expected=torch.stack((self.values[0],torch.quantile(self.values,.5,dim=0),self.values[-1]))
        torch.testing.assert_close(out[:,4:],expected,rtol=0,atol=0)

    def test_checkpoint_restores_source_without_original_file(self):
        source=AttributeQuantileSource(dict(self.options,shape=True))
        state=copy.deepcopy(source.state_dict());self.path.unlink()
        restored=AttributeQuantileSource(dict(path=None,knots=13,shape=True))
        with self.assertRaises(RuntimeError):restored(torch.zeros(1,7),torch.zeros(1,7))
        restored.load_state_dict(state,strict=True)
        z=torch.randn(30,7)
        self.assertTrue(torch.equal(source(z,z),restored(z,z)))
        self.assertEqual(restored.source_sha256,self.options['sha256'])

    def test_hash_mismatch_and_changed_fields_fail(self):
        with self.assertRaises(ValueError):AttributeQuantileSource(dict(self.options,sha256='0'*64))
        a=AttributeQuantileSource(self.options)
        b=AttributeQuantileSource(dict(self.options,shape=True))
        with self.assertRaises(ValueError):b.load_state_dict(a.state_dict())
        state=copy.deepcopy(a.state_dict());state['quantile_values'][5,0]=-2.
        with self.assertRaises(RuntimeError):a.load_state_dict(state)

    def test_empirical_stationary_mass_from_existing_gaussian_draws(self):
        source=AttributeQuantileSource(self.options)
        generator=torch.Generator().manual_seed(3917)
        z=torch.randn(100000,7,generator=generator)
        speed=source(z,z)[:,6]
        # Two equal zero-speed observations occupy u in [0, 1/3] under linear
        # interpolation of four empirical observations, unlike a hard atom CDF.
        self.assertAlmostEqual(float((speed==0).float().mean()),1/3,delta=.008)

    def wrapper(self,source=None):
        from src.smart.diffusion.initial_diffusion import InitDiffusion
        processor=SimpleNamespace(use_refiner=False,learn_init=True,init_map_range=50.)
        args=SimpleNamespace(input_dim=7,hidden_dim=32,num_heads=4,dropout=0.,num_denoiser_layers=1,
            num_branch_steps=1,branch_steps=[0],sampling_steps=20,use_rl=False,sigma_h=1.)
        with patch.object(InitDiffusion,'_make_args',return_value=args):
            return InitDiffusion(32,4,4,processor,False,velocity_representation='speed',
                denoiser_num_layers=1,attribute_source=source)

    def test_option_preserves_network_weights_and_legacy_checkpoints(self):
        torch.manual_seed(97);legacy=self.wrapper();after=torch.get_rng_state().clone()
        state=copy.deepcopy(legacy.state_dict());state['_extra_state']['denoiser_options'].pop('attribute_source')
        self.wrapper().load_state_dict(state,strict=True)
        torch.manual_seed(97);candidate=self.wrapper(self.options)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for name,value in legacy.G1.named_parameters():
            torch.testing.assert_close(value,dict(candidate.G1.named_parameters())[name],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in legacy.parameters()),sum(p.numel() for p in candidate.parameters()))
        state=copy.deepcopy(candidate.state_dict())
        self.wrapper(dict(self.options,path=None)).load_state_dict(state,strict=True)
        with self.assertRaisesRegex(ValueError,'denoiser options'):
            self.wrapper().load_state_dict(state,strict=True)

    def test_training_and_generation_share_the_source_hook(self):
        flow=self.wrapper(self.options).G1
        state=torch.tensor([[0.,0.,1.,0.,4.,2.,3.],[2.,1.,1.,0.,4.,2.,4.],[4.,2.,1.,0.,4.,2.,5.]])
        flow.model.normal_mean.fill_(1.)
        flow.model.normal_scale.fill_(1.)
        agent=dict(batch=torch.zeros(3,dtype=torch.long),type=torch.zeros(3,dtype=torch.long),
            num_graphs=1,ego_mask=torch.tensor([True,False,False]),expert_input=state.clone())
        source=flow.model.attribute_source
        with patch.object(source,'forward',wraps=source.forward) as draw:
            flow._sample_noise(state,copy.deepcopy(agent))
            self.assertEqual(draw.call_count,1)
            flow.eval()
            with patch.object(flow.model,'forward',return_value=state.clone()):
                flow.sample(copy.deepcopy(agent),{},steps=20)
            self.assertEqual(draw.call_count,2)


if __name__=='__main__':unittest.main()
