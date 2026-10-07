"""Flow sampling keeps fixed context at every field evaluation.

A synthetic vector field makes intermediate condition drift observable without
large weights. The legacy predictor clone is an in-memory diagnostic only; the
production sampler always projects fixed nodes before the Heun corrector.
"""
from __future__ import annotations

import ast
import copy
import inspect
import textwrap
import unittest

import torch
from torch_geometric.data import HeteroData

from src.smart.vectorworld.core import FlowLDM


def _graph(*, outside_clip=False):
    data = HeteroData()
    for kind in ("agent", "lane"):
        data[kind].num_nodes = 2
        data[kind].batch = torch.zeros(2, dtype=torch.long)
        values = [[12., 16.], [-12., -16.]] if outside_clip else [[.2, .4], [-.2, -.4]]
        data[kind].latents = torch.tensor(values)
        data[kind].x = data[kind].latents.clone()
    data['agent'].partition_mask = data['agent'].mask = torch.tensor([True, False])
    data['lane'].partition_mask = data['lane'].mask = torch.tensor([False, True])
    data.batch_size = 1
    return data


class _SyntheticField:
    """Agent updates depend on lane context, with nonzero lane velocity."""

    n_steps = 2

    def __init__(self, *, solver="heun", clip=None):
        self.flow_solver, self.diffusion_clip = solver, clip
        self.agent_inputs, self.lane_inputs = [], []

    def _cfg_vector_field(self, x_agent, x_lane, data, t_agent, t_lane):
        self.agent_inputs.append(x_agent.detach().clone())
        self.lane_inputs.append(x_lane.detach().clone())
        return torch.ones_like(x_agent) * (1. + x_lane.mean()), torch.full_like(x_lane, 2.)


def _legacy_predictor_sampler():
    """Remove only predictor projections to compare the previous semantics."""
    function = copy.deepcopy(ast.parse(textwrap.dedent(inspect.getsource(FlowLDM.sample))).body[0])
    function.name = "legacy_predictor_sample"

    class RemovePredictorProjection(ast.NodeTransformer):
        removed = 0

        def visit_If(self, node):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Assign):
                target = node.body[0].targets[0]
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id in ("x_agent_tmp", "x_lane_tmp")):
                    self.removed += 1
                    return None
            return self.generic_visit(node)

    transformer = RemovePredictorProjection()
    function = transformer.visit(function)
    if transformer.removed != 2:
        raise AssertionError("Expected exactly two fixed-context predictor projections")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, "<legacy-predictor-audit>", "exec"), namespace)
    return namespace[function.name]


def _sample(mode, *, method=FlowLDM.sample, solver="heun", clip=None, outside_clip=False):
    data = _graph(outside_clip=outside_clip)
    field = _SyntheticField(solver=solver, clip=clip)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        output = method(
            field, data['agent'].x[:, None, :].shape,
            data['lane'].x[:, None, :].shape, data,
            device="cpu", mode=mode,
        )
    return data, field, output


class VectorWorldFlowConditioningTest(unittest.TestCase):
    def test_lane_condition_is_fixed_at_every_heun_field_evaluation(self):
        data, field, (_, lanes) = _sample("lane_conditioned")
        fixed = data['lane'].latents[:, None, :]
        self.assertEqual(len(field.lane_inputs), 4)
        for value in field.lane_inputs:
            torch.testing.assert_close(value, fixed, rtol=0, atol=0)
        torch.testing.assert_close(lanes, data['lane'].latents, rtol=0, atol=0)

    def test_projection_prevents_legacy_predictor_drift_and_changes_agent_update(self):
        _, legacy_field, (legacy_agents, legacy_lanes) = _sample(
            "lane_conditioned", method=_legacy_predictor_sampler()
        )
        data, _, (agents, lanes) = _sample("lane_conditioned")
        fixed = data['lane'].latents[:, None, :]
        deviations = [float((value - fixed).abs().max()) for value in legacy_field.lane_inputs]
        torch.testing.assert_close(torch.tensor(deviations), torch.tensor([0., 1., 0., 1.]))
        torch.testing.assert_close(lanes, legacy_lanes, rtol=0, atol=0)
        torch.testing.assert_close(legacy_agents - agents, torch.full_like(agents, .5))

    def test_joint_generation_remains_exactly_equal_to_legacy_predictor(self):
        _, legacy_field, legacy_output = _sample("initial_scene", method=_legacy_predictor_sampler())
        _, field, output = _sample("initial_scene")
        for legacy, current in zip(legacy_field.agent_inputs + legacy_field.lane_inputs,
                                   field.agent_inputs + field.lane_inputs):
            torch.testing.assert_close(current, legacy, rtol=0, atol=0)
        for legacy, current in zip(legacy_output, output):
            torch.testing.assert_close(current, legacy, rtol=0, atol=0)

    def test_selective_agent_and_lane_conditions_survive_heun_and_clipping(self):
        for mode in ("train", "inpainting"):
            for outside_clip in (False, True):
                with self.subTest(mode=mode, outside_clip=outside_clip):
                    data, field, output = _sample(mode, clip=.5, outside_clip=outside_clip)
                    self.assertEqual(len(field.lane_inputs), 4)
                    self.check_selective_conditions(data, field, output)

    def test_euler_conditions_are_fixed_at_every_call_and_after_clipping(self):
        for mode in ("lane_conditioned", "train", "inpainting"):
            with self.subTest(mode=mode):
                data, field, output = _sample(mode, solver="euler", clip=.5, outside_clip=True)
                self.assertEqual(len(field.lane_inputs), 2)
                if mode == "lane_conditioned":
                    for value in field.lane_inputs:
                        torch.testing.assert_close(value[:, 0], data['lane'].latents, rtol=0, atol=0)
                    torch.testing.assert_close(output[1], data['lane'].latents, rtol=0, atol=0)
                else:
                    self.check_selective_conditions(data, field, output)

    def check_selective_conditions(self, data, field, output):
        for kind, inputs, generated in zip(("agent", "lane"),
                                           (field.agent_inputs, field.lane_inputs), output):
            mask = data[kind].mask
            fixed = data[kind].latents[mask]
            for value in inputs:
                torch.testing.assert_close(value[mask, 0], fixed, rtol=0, atol=0)
            torch.testing.assert_close(generated[mask], fixed, rtol=0, atol=0)
            # Unconditioned nodes continue to evolve rather than being frozen.
            self.assertFalse(torch.equal(generated[~mask], data[kind].latents[~mask]))
            self.assertLessEqual(float(generated[~mask].abs().max()), .5)


if __name__ == "__main__":
    unittest.main()
