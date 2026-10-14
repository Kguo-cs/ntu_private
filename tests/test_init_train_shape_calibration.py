"""Meaningful invariants for the isolated TRAIN-only size-calibration experiment."""
import types
import unittest
import torch
from tools.init_train_shape_calibration import fit_transport, apply_transport, install_transport


def data():
    x = torch.linspace(.5, 8, 1024, dtype=torch.float64)
    shape = torch.stack([x, x*.4], -1)
    kind = torch.zeros(len(x), dtype=torch.long)
    ego = torch.zeros_like(kind, dtype=torch.bool)
    return shape, kind, ego


class ShapeCalibrationTest(unittest.TestCase):
    def test_known_scale_map_and_extrapolation_are_positive_monotone(self):
        shape, kind, ego = data()
        fit = fit_transport(shape, kind, ego, shape*1.7, kind, ego)
        values = torch.tensor([[.01,.005],[1.,.4],[6.,2.4],[100.,40.]], dtype=shape.dtype)
        result = apply_transport(values, kind[:4], ego[:4], fit)
        torch.testing.assert_close(result, values*1.7, rtol=1e-10, atol=1e-10)
        assert torch.all(result[1:] > result[:-1])


    def test_identity_preserves_all_inputs_and_consumes_no_rng(self):
        shape, kind, ego = data()
        fit = fit_transport(shape, kind, ego, shape, kind, ego)
        before = torch.random.get_rng_state().clone()
        torch.testing.assert_close(apply_transport(shape, kind, ego, fit), shape, rtol=1e-12, atol=1e-12)
        assert torch.equal(before, torch.random.get_rng_state())


    def test_ego_nonvehicles_and_invalid_predictions_are_not_overwritten(self):
        shape, kind, ego = data()
        fit = fit_transport(shape, kind, ego, shape*2, kind, ego)
        inputs = torch.tensor([[4.,2.],[3.,1.],[2.,1.],[-1.,1.],[5.,2.]])
        original = inputs.clone()
        output = apply_transport(inputs, torch.tensor([0,1,2,0,0]), torch.tensor([1,0,0,0,0]), fit)
        assert torch.equal(output[:4], inputs[:4]) and torch.equal(inputs, original)
        torch.testing.assert_close(output[-1], inputs[-1]*2)


    def test_duplicate_quantiles_are_monotone_and_complete_collapse_is_rejected(self):
        shape, kind, ego = data()
        shape[:600] = shape[600]
        fit = fit_transport(shape, kind, ego, data()[0], kind, ego)
        curve = fit['curves'][0]
        assert all(a < b for a,b in zip(curve['x'],curve['x'][1:]))
        assert all(a <= b for a,b in zip(curve['y'],curve['y'][1:]))
        with self.assertRaisesRegex(ValueError, 'Collapsed'):
            fit_transport(torch.ones_like(shape), kind, ego, shape, kind, ego)


    def test_wrapper_changes_real_boxes_but_preserves_other_outputs_and_ego(self):
        shape, kind, ego = data()
        fit = fit_transport(shape, kind, ego, shape*2, kind, ego)
        ego[0] = True
        sentinels = [object() for _ in range(4)]
        decoder = types.SimpleNamespace(training=False)
        decoder._infer = lambda agent, feature: (sentinels[0],sentinels[1],sentinels[2],shape,sentinels[3])
        model = types.SimpleNamespace(encoder=types.SimpleNamespace(initial_scene_only=True,init_decoder_name='flow',init_decoder=decoder))
        stats = install_transport(model, fit)
        output = decoder._infer({'type':kind,'ego_mask':ego}, {})
        assert output[0] is sentinels[0] and output[1] is sentinels[1] and output[2] is sentinels[2] and output[4] is sentinels[3]
        torch.testing.assert_close(output[3][1:], shape[1:]*2)
        assert torch.equal(output[3][0], shape[0]) and stats['changed_vehicles']==1023
        decoder.training=True
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            decoder._infer({'type':kind,'ego_mask':ego},{})


    def test_eval_fitted_artifact_rejected(self):
        shape, kind, ego = data()
        fit = fit_transport(shape,kind,ego,shape,kind,ego)
        fit['evaluation_data_used_for_fit']=True
        with self.assertRaises(ValueError):apply_transport(shape,kind,ego,fit)


if __name__ == '__main__':
    unittest.main()
