"""Strict map-only initialization must preserve fresh denoiser/EMA state."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from lightning import LightningModule

from src.smart.model.smart import SMART
from src.smart.model.smart_gail import SMART_GAIL
import test_init_diffusion_sep_map as sep_map_fixtures


class FixedMapCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "map.ckpt"
        self.model = SMART.__new__(SMART)
        LightningModule.__init__(self.model)
        encoder = torch.nn.Module()
        encoder.map_encoder = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.7))
        encoder.agent_encoder = torch.nn.Linear(3, 4).requires_grad_(False)
        encoder.init_decoder = torch.nn.Module()
        encoder.init_decoder.G1 = torch.nn.Linear(4, 2)
        encoder.init_decoder.ema = {"num_updates": 0, "shadow_params": [torch.randn(2, 4)]}
        encoder.init_decoder_name = "flow"
        encoder.initial_scene_only, encoder.gail, encoder.sep_map = True, False, False
        self.model.encoder = encoder
        self.model.token_processor = SimpleNamespace(learn_init=True)
        self.model.initial_map_checkpoint_metadata = None
        self.model.lr, self.model.lr_total_steps = .0005, 128
        self.model.lr_warmup_steps, self.model.lr_min_ratio = 0, .05
        self.model.optimizer_profile = "default"
        self.source = {key: value.detach().clone() + 1
                       for key, value in encoder.map_encoder.state_dict().items()}

    def save(self, source=None):
        state = {"encoder.map_encoder." + key: value for key, value in
                 (self.source if source is None else source).items()}
        # This bait must never overwrite the fresh generator.
        state["encoder.init_decoder.G1.weight"] = torch.zeros(2, 4)
        torch.save({"state_dict": state, "epoch": 9, "global_step": 152190}, self.path)

    def test_loads_only_map_and_keeps_generator_ema_fresh(self):
        self.save()
        before = copy.deepcopy(self.model.encoder.init_decoder.state_dict())
        ema = copy.deepcopy(self.model.encoder.init_decoder.ema)
        self.model._initialize_fixed_map_encoder(self.path, True)
        for key, value in self.source.items():
            self.assertTrue(torch.equal(value, self.model.encoder.map_encoder.state_dict()[key]))
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.model.encoder.init_decoder.state_dict()[key]))
        self.assertEqual(self.model.encoder.init_decoder.ema["num_updates"], 0)
        self.assertTrue(torch.equal(ema["shadow_params"][0],
                                    self.model.encoder.init_decoder.ema["shadow_params"][0]))
        self.assertEqual(self.model.initial_map_checkpoint_metadata["loaded_tensor_count"], 2)
        self.assertEqual(self.model.initial_map_checkpoint_metadata["source_global_step"], 152190)
        self.assertEqual(len(self.model.initial_map_checkpoint_metadata["file_sha256"]), 64)

    def test_frozen_map_stays_in_eval_and_optimizer_excludes_it(self):
        self.save()
        self.model._initialize_fixed_map_encoder(self.path, True)
        self.model.train()
        self.assertTrue(self.model.encoder.init_decoder.training)
        self.assertFalse(self.model.encoder.map_encoder.training)
        self.assertFalse(self.model.encoder.map_encoder[1].training)
        self.assertFalse(any(p.requires_grad for p in self.model.encoder.map_encoder.parameters()))
        optimizers, _ = SMART_GAIL.configure_optimizers(self.model)
        optimized = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
        self.assertFalse(optimized.intersection(id(p) for p in self.model.encoder.map_encoder.parameters()))
        self.assertEqual(optimized, {id(p) for p in self.model.encoder.init_decoder.parameters()})
        frozen = copy.deepcopy(self.model.encoder.map_encoder.state_dict())
        optimizers[0].zero_grad()
        self.model.encoder.init_decoder.G1(torch.ones(1, 4)).sum().backward()
        optimizers[0].step()
        for key, value in frozen.items():
            self.assertTrue(torch.equal(value, self.model.encoder.map_encoder.state_dict()[key]))

    def test_default_null_preserves_parent_training_mode(self):
        self.model.train()
        self.assertTrue(self.model.encoder.map_encoder.training)
        self.assertTrue(self.model.encoder.map_encoder[1].training)

    def test_pretrained_map_seeds_trainable_separate_map_and_resets_its_ema(self):
        fixture = sep_map_fixtures.InitDiffusionSeparateMapTest()
        for overrides in ({}, {'hidden_dim': 12, 'num_layers': 2}):
            with self.subTest(overrides=overrides):
                decoder = fixture.decoder(init_map_encoder_config=overrides,
                                          init_diffusion={'use_ema': True, 'ema_decay': .9})
                model = fixture.training_model(decoder)
                model.initial_map_checkpoint_metadata = None
                before = copy.deepcopy(decoder.init_map_encoder.state_dict())
                generator = copy.deepcopy(decoder.init_decoder.state_dict())
                source = {key: value.detach().clone() + .5 for key, value in
                          decoder.map_encoder.state_dict().items()}
                torch.save({'state_dict': {'encoder.map_encoder.' + k: v for k, v in source.items()}},
                           self.path)
                model._initialize_fixed_map_encoder(self.path, True)
                model._configure_finetuning(True)
                model.train()
                self.assertFalse(decoder.map_encoder.training)
                self.assertFalse(any(p.requires_grad for p in decoder.map_encoder.parameters()))
                self.assertTrue(all(p.requires_grad for p in decoder.init_map_encoder.parameters()))
                for name, value in decoder.init_map_encoder.state_dict().items():
                    expected = (source[name] if name in source and source[name].shape == value.shape
                                else before[name])
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                for shadow, parameter in zip(decoder.initial_map_ema.shadow_params,
                                             decoder.init_map_encoder.parameters()):
                    torch.testing.assert_close(shadow, parameter, rtol=0, atol=0)
                self.assertEqual(decoder.initial_map_ema.num_updates, 0)
                for name, value in decoder.init_decoder.state_dict().items():
                    torch.testing.assert_close(value, generator[name], rtol=0, atol=0)

    def test_missing_or_extra_map_keys_fail_without_partial_copy(self):
        for mutation in (lambda s: s.pop("0.bias"), lambda s: s.update(extra=torch.ones(1))):
            with self.subTest(mutation=mutation):
                source = copy.deepcopy(self.source)
                mutation(source)
                self.save(source)
                before = copy.deepcopy(self.model.encoder.map_encoder.state_dict())
                with self.assertRaisesRegex(ValueError, "keys do not match"):
                    self.model._initialize_fixed_map_encoder(self.path, True)
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, self.model.encoder.map_encoder.state_dict()[key]))

    def test_wrong_shape_dtype_or_nonfinite_fail_before_any_copy(self):
        # Keep the first weight valid and modify the second tensor so an
        # unchecked sequential copy could have mutated the first parameter.
        for bad in (torch.ones(5), torch.ones(4, dtype=torch.float64),
                    torch.full((4,), float("nan")), torch.full((4,), float("inf")), [1.] * 4):
            with self.subTest(bad=bad):
                source = copy.deepcopy(self.source)
                source["0.bias"] = bad
                self.save(source)
                before = copy.deepcopy(self.model.encoder.map_encoder.state_dict())
                with self.assertRaisesRegex(ValueError, "tensor"):
                    self.model._initialize_fixed_map_encoder(self.path, True)
                for key, value in before.items():
                    self.assertTrue(torch.equal(value, self.model.encoder.map_encoder.state_dict()[key]))

    def test_rejects_wrong_training_scope(self):
        self.save()
        for attribute, value in (("gail", True),
                                 ("initial_scene_only", False), ("init_decoder_name", "scenario_dreamer")):
            with self.subTest(attribute=attribute):
                old = getattr(self.model.encoder, attribute)
                setattr(self.model.encoder, attribute, value)
                with self.assertRaisesRegex(ValueError, "requires supervised"):
                    self.model._initialize_fixed_map_encoder(self.path, True)
                setattr(self.model.encoder, attribute, old)
        self.model.token_processor.learn_init = False
        with self.assertRaisesRegex(ValueError, "requires supervised"):
            self.model._initialize_fixed_map_encoder(self.path, True)
        self.model.token_processor.learn_init = True
        with self.assertRaisesRegex(ValueError, "requires supervised"):
            self.model._initialize_fixed_map_encoder(self.path, False)

    def test_rejects_missing_file_and_malformed_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            self.model._initialize_fixed_map_encoder(self.path, True)
        for checkpoint in ({}, {"state_dict": []}, [1]):
            torch.save(checkpoint, self.path)
            with self.assertRaisesRegex(ValueError, "state_dict mapping"):
                self.model._initialize_fixed_map_encoder(self.path, True)


if __name__ == "__main__":
    unittest.main()
