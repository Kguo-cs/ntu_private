import unittest

import torch

from src.smart.diffusion.diffusion_utils import (
    get_col_rate,
    get_diff_loss,
    multi_circle_collision_loss_mem_efficient,
)


def states(y_coordinates):
    result = torch.zeros(len(y_coordinates), 8)
    result[:, 1] = torch.tensor(y_coordinates, dtype=torch.float32)
    result[:, 2] = 1.0
    result[:, 4:6] = torch.tensor([4.5, 2.0])
    return result


def supervised_loss(prediction, reference, batch=None, time=None, use_match=False):
    n = len(prediction)
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long)
    if time is None:
        time = torch.full((n, 1), 0.5)
    agent = {'batch': batch, 'type': torch.ones(n, dtype=torch.long)}
    return get_diff_loss(
        agent, prediction, reference, time, t_eps=0.05,
        use_col=True, x_pred=True, use_match=use_match, all_state=True,
    )[1]


class CollisionGTToleranceTest(unittest.TestCase):
    def test_exact_gt_overlap_has_zero_supervised_penalty_and_gradient(self):
        reference = states([0.0, 1.4])
        prediction = reference.clone().requires_grad_(True)
        loss = supervised_loss(prediction, reference)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

        # Actual collision reporting must still count these overlapping agents.
        agent = {'batch': torch.zeros(2, dtype=torch.long)}
        torch.testing.assert_close(get_col_rate(agent, reference), torch.zeros(2))

    def test_existing_overlap_can_improve_but_cannot_deepen_for_free(self):
        reference = states([0.0, 1.4])
        for separation in (1.4, 1.8, 3.0):
            with self.subTest(separation=separation):
                loss = supervised_loss(states([0.0, separation]), reference)
                self.assertEqual(loss.item(), 0.0)
        shallow = supervised_loss(states([0.0, 1.2]), reference)
        deep = supervised_loss(states([0.0, 1.0]), reference)
        self.assertGreater(shallow.item(), 0.0)
        self.assertGreater(deep.item(), shallow.item())

    def test_new_collision_keeps_original_penalty_for_all_agent_types(self):
        reference = states([0.0, 3.0])
        prediction = states([0.0, 1.4])
        batch = torch.zeros(2, dtype=torch.long)
        raw, _, _ = multi_circle_collision_loss_mem_efficient(prediction, batch)
        for types in ([0, 0], [0, 1], [1, 2]):
            with self.subTest(types=types):
                agent = {'batch': batch, 'type': torch.tensor(types)}
                loss = get_diff_loss(
                    agent, prediction, reference, torch.full((2, 1), 0.5),
                    t_eps=0.05, use_col=True, x_pred=True,
                )[1]
                self.assertGreater(loss.item(), 0.0)
                torch.testing.assert_close(loss, raw.mean() * 2.0)

    def test_tolerance_is_pair_specific_and_does_not_cross_scenes(self):
        reference = states([0.0, 1.4, 0.0, 3.0])
        prediction = states([0.0, 1.4, 0.0, 1.4])
        batch = torch.tensor([0, 0, 3, 3])
        penalty, end, start = multi_circle_collision_loss_mem_efficient(
            prediction, batch, reference_state=reference
        )
        self.assertEqual(list(zip(start.tolist(), end.tolist())), [(0, 1), (2, 3)])
        self.assertEqual(penalty[0].item(), 0.0)
        self.assertGreater(penalty[1].item(), 0.0)

    def test_gt_tolerance_is_detached_and_prediction_position_is_differentiable(self):
        reference = states([0.0, 1.4]).requires_grad_(True)
        prediction = states([0.0, 1.0]).requires_grad_(True)
        loss = supervised_loss(prediction, reference)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        # Gradient descent separates the two agents instead of shrinking them.
        self.assertGreater(prediction.grad[0, 1].item(), 0.0)
        self.assertLess(prediction.grad[1, 1].item(), 0.0)
        torch.testing.assert_close(prediction.grad[:, 4:6], torch.zeros(2, 2))

    def test_fixed_ego_order_does_not_change_penalty_or_movable_gradient(self):
        reference = states([0.0, 3.0])
        prediction = states([0.0, 1.4]).requires_grad_(True)
        time = torch.tensor([[0.0], [0.5]])
        loss = supervised_loss(prediction, reference, time=time)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

        reversed_prediction = prediction.detach().flip(0).requires_grad_(True)
        reversed_loss = supervised_loss(
            reversed_prediction, reference.flip(0), time=time.flip(0)
        )
        reversed_loss.backward()
        torch.testing.assert_close(loss, reversed_loss)
        torch.testing.assert_close(prediction.grad[1], reversed_prediction.grad[0])

    def test_single_agent_scenes_return_differentiable_zero(self):
        for count in (1, 3):
            with self.subTest(count=count):
                reference = states(list(range(count)))
                prediction = reference.clone().requires_grad_(True)
                loss = supervised_loss(
                    prediction, reference, batch=torch.arange(count),
                    time=torch.zeros(count, 1),
                )
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_reference_pairs_follow_matching_result(self):
        reference = states([0.0, 1.4, 10.0])
        prediction = reference[[2, 0, 1]].clone().requires_grad_(True)
        loss = supervised_loss(prediction, reference, use_match=True)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


if __name__ == '__main__':
    unittest.main()
