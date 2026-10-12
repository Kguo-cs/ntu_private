import unittest

import torch

from src.smart.diffusion.distribution_loss import AttributeDistributionLoss


class DistributionLossTests(unittest.TestCase):
    def fixture(self):
        target = torch.tensor([[0., 0., 1., 0., 4., 2., 0.],
                               [0., 0., 1., 0., 5., 2.2, 10.],
                               [0., 0., 1., 0., 2., 1., 5.]])
        return target, torch.full((3, 1), .8), torch.ones(3, 7, dtype=torch.bool), torch.tensor([0, 0, 1]), torch.ones(1, 7)

    def test_transport_is_order_invariant_within_type(self):
        target, time, valid, labels, scale = self.fixture()
        pred = target[[1, 0, 2]].clone().requires_grad_()
        loss, _ = AttributeDistributionLoss({'speed': 1., 'length': 1.})(pred, target, time, valid, labels, scale)
        self.assertEqual(loss.item(), 0.)

    def test_collapsed_prediction_has_finite_corrective_gradient(self):
        target, time, valid, labels, scale = self.fixture()
        pred = target.clone(); pred[:, 6] = 5.; pred.requires_grad_()
        loss, _ = AttributeDistributionLoss({'speed': 2.})(pred, target, time, valid, labels, scale)
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[0, 6].item(), 0.)
        self.assertLess(pred.grad[1, 6].item(), 0.)

    def test_masks_fixed_low_noise_and_invalid_targets(self):
        target, time, valid, labels, scale = self.fixture()
        target[0, 6] = float('nan'); time[1] = .1
        pred = torch.nan_to_num(target).requires_grad_()
        loss, _ = AttributeDistributionLoss({'speed': 1.})(pred, target, time, valid, labels, scale)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_config_rejected(self):
        for cfg in ({'speed': -1}, {'time_min': 1}, {'kind': 'ce'}, {'mystery': 1}):
            with self.assertRaises(ValueError):
                AttributeDistributionLoss(cfg)


if __name__ == '__main__':
    unittest.main()
