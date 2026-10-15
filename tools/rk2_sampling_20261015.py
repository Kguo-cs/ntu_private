"""Process-local two-stage RK2 on Euclidean state and circular heading."""
from types import MethodType
import torch
from src.smart.utils import wrap_angle


@torch.no_grad()
def rk2_step(self, original, latent, start, end, agent, feature, stage_fraction=2/3):
    c = float(stage_fraction)
    if not 0 < c < 1:
        raise ValueError('RK2 internal stage must lie strictly inside its interval')
    middle = start + c*(end-start)
    if float(middle) < self.t_eps-1e-7:
        raise ValueError('RK2 model evaluation is below the native velocity floor')
    b2, b1 = 1/(2*c), 1-1/(2*c)
    alpha, beta = b1/c, b2/(1-c)
    type_initial = (agent['_init_diffusion_type_state'].clone()
                    if self.generate_type and not self.joint_type else None)
    first = original(latent, start, middle, agent, feature, branch_mask=None)
    predictor = first[0]
    type_middle = (agent['_init_diffusion_type_state'].clone() if type_initial is not None else None)
    second = original(predictor, middle, end, agent, feature, branch_mask=None)
    result = latent + alpha*(predictor-latent) + beta*(second[0]-predictor)
    if self.heading_noise == 'circular':
        theta = torch.atan2(latent[:,3], latent[:,2])
        stage_theta = torch.atan2(predictor[:,3], predictor[:,2])
        end_theta = torch.atan2(second[0][:,3], second[0][:,2])
        theta = wrap_angle(theta + alpha*wrap_angle(stage_theta-theta) + beta*wrap_angle(end_theta-stage_theta))
        result = torch.cat((result[:,:2],torch.stack((theta.cos(),theta.sin()),-1),result[:,4:]),-1)
    fixed = self._conditioned_state_mask(agent,result)
    clean = self._joint_clean(agent['expert_input'],agent)
    result = torch.where(fixed,clean,result)
    if type_initial is not None:
        state = type_initial + alpha*(type_middle-type_initial) + beta*(agent['_init_diffusion_type_state']-type_middle)
        mask = self._conditioned_type_mask(agent)
        state[mask] = agent['_init_diffusion_type_source'][mask]
        agent['_init_diffusion_type_state'] = state
    if self.joint_type:
        agent['_init_diffusion_type_state'] = result[:,self.model.m_delta_dim:]
    return (result,second[1],first[2],second[3],second[4],second[5])


def install_ralston(flow):
    if flow.use_sde or flow.sampling_exact_x0 or flow.sampling_integrator != 'euler':
        raise ValueError('Experimental Ralston requires the deterministic native Euler base')
    if flow.collision_guidance['scale']:
        raise ValueError('This experiment excludes substep collision projections')
    if getattr(flow,'_experimental_ralston_installed',False):
        raise ValueError('Ralston is already installed')
    function = flow._sample_step.__func__
    @torch.no_grad()
    def step(self, latent, start, end, agent, feature, branch_mask=None):
        if branch_mask is not None:
            raise ValueError('Ralston does not support stochastic branches')
        return rk2_step(self,function.__get__(self,type(self)),latent,start,end,agent,feature)
    flow._sample_step = MethodType(step,flow)
    flow._experimental_ralston_installed = True
