"""TEST-only position velocity floor experiment; no trained state is modified."""
from __future__ import annotations
import copy
from pathlib import Path
from tools.evaluate_init_diffusion_sampler import InferenceSamplerHookCore, require, save


def hybrid_position_velocity(latent, time, velocity, x0, *, enabled=True):
    """Return the original velocity except position at0<t<.05, with floor1e-4."""
    import torch
    require(latent.ndim == velocity.ndim == x0.ndim == 2 and latent.shape == velocity.shape == x0.shape
            and latent.shape[1] >= 7 and time.shape == (latent.shape[0], 1), 'Expected matching[N,7+] state and[N,1]time')
    low = (time[:, 0] > 0) & (time[:, 0] < .05)
    if not enabled or not bool(low.any()):
        return velocity, low
    corrected = (latent[:, :2] - x0[:, :2]) / time.clamp_min(1e-4)
    pos = torch.where(low[:, None], corrected, velocity[:, :2])
    return torch.cat((pos, velocity[:, 2:]), dim=-1), low


class HybridPositionHookCore(InferenceSamplerHookCore):
    def __init__(self, *, hybrid_position=False, **kwargs):
        super().__init__(**kwargs)
        require(self.steps == 40 and self.eps == .05, 'Hybrid experiment requires unchangedN40/basefloor.05')
        require(isinstance(hybrid_position, bool), 'hybrid_position must be boolean')
        self.hybrid_position = hybrid_position
        self.record.update(hybrid_position=hybrid_position, hybrid_position_floor=1e-4,
            hybrid_position_time_window='0<t<0.05', unchanged_base_floor=.05,
            prediction_velocity_calls=0, lowtime_velocity_calls=0, hybrid_active_calls=0,
            nonposition_velocity_preserved=True, x0_preserved=True,
            prediction_velocity_wrapper_restored=False, GT_postprocessing_added=False)

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        flow = pl_module.encoder.init_decoder.G1
        require(flow.heading_noise == 'circular' and flow.heading_objective == 'x0'
                and flow.velocity_representation == 'speed' and flow.size_representation == 'linear'
                and flow.type_process == 'separate', 'Hybrid cohort source representation changed')
        self.original_prediction_velocity = flow._prediction_velocity
        self.method_was_instance_attribute = '_prediction_velocity' in flow.__dict__
        self.old_method_attribute = flow.__dict__.get('_prediction_velocity')
        def observed_prediction_velocity(latent, time, prediction, *args, **kwargs):
            import torch
            require(not flow.training, 'Position velocity override is TEST-only')
            velocity, x0 = self.original_prediction_velocity(latent, time, prediction, *args, **kwargs)
            self.record['prediction_velocity_calls'] += 1
            changed, low = hybrid_position_velocity(latent, time, velocity, x0, enabled=self.hybrid_position)
            if bool(low.any()):
                self.record['lowtime_velocity_calls'] += 1
                if self.hybrid_position:
                    self.record['hybrid_active_calls'] += 1
                    require(torch.equal(changed[:, 2:], velocity[:, 2:]), 'Hybrid changed non-position velocity')
            return changed, x0
        flow._prediction_velocity = observed_prediction_velocity
        save(self.audit_path, self.record)

    def on_test_end(self, trainer, pl_module):
        flow = pl_module.encoder.init_decoder.G1
        require(self.record['prediction_velocity_calls'] == self.steps * self.record['sample_calls'], 'Unexpected prediction velocity call count')
        require(self.record['lowtime_velocity_calls'] == self.record['sample_calls'], 'UniformN40 must contain one lowtime call per sample')
        require(self.record['hybrid_active_calls'] == (self.record['sample_calls'] if self.hybrid_position else 0), 'Actual hybrid mode/call count differs')
        if self.method_was_instance_attribute:
            flow._prediction_velocity = self.old_method_attribute
        else:
            del flow._prediction_velocity
        require(flow._prediction_velocity == self.original_prediction_velocity, 'Prediction velocity method not restored')
        self.record['prediction_velocity_wrapper_restored'] = True
        self.record['unchanged_EMA_updates'] = pl_module.encoder.init_decoder.ema.num_updates
        super().on_test_end(trainer, pl_module)


def make_hybrid_hook(**kwargs):
    import lightning as L
    class Hook(HybridPositionHookCore, L.Callback):
        pass
    return Hook(**kwargs)
