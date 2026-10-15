"""RK2 order, circular geometry, type/ego preservation and call-count checks."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/sim_mpl_continuous'
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import copy
    import datetime
    import json
    import torch
    from src.smart.diffusion.midpoint_sampling import midpoint_step
    from tools.rk2_sampling_20261015 import rk2_step,install_ralston

    torch.set_num_threads(1)
    class Flow:
        t_eps=0.
        generate_type=True
        joint_type=False
        heading_noise='circular'
        use_sde=False
        sampling_exact_x0=False
        sampling_integrator='euler'
        collision_guidance={'scale':0}
        def __init__(self): self.calls=0
        def _conditioned_state_mask(self,agent,state):
            mask=torch.zeros_like(state,dtype=torch.bool)
            mask[0]=True
            return mask
        def _conditioned_type_mask(self,agent): return torch.tensor([True,False])
        def _joint_clean(self,clean,agent): return clean
        def _sample_step(self,state,start,end,agent,feature,branch_mask=None):
            self.calls+=1
            dt=end-start
            result=state.clone()
            result[:,0]+=dt*state[:,0].square()
            theta=torch.atan2(state[:,3],state[:,2])+dt*1.3
            result[:,2:4]=torch.stack((theta.cos(),theta.sin()),-1)
            fixed=self._conditioned_state_mask(agent,result)
            result=torch.where(fixed,agent['expert_input'],result)
            agent['_init_diffusion_type_state']+=dt*state.new_tensor([.3,-.2,.1])
            agent['_init_diffusion_type_state'][0]=agent['_init_diffusion_type_source'][0]
            return (result,None,None,None,None,None)
    state=torch.tensor([[.2,0,1,0,4,2,1],[.4,0,1,0,4,2,1]],dtype=torch.float64)
    def agent():
        return {'expert_input':state.clone(),
                '_init_diffusion_type_source':torch.tensor([[1.,0,0],[0,1.,0]],dtype=torch.float64),
                '_init_diffusion_type_state':torch.tensor([[1.,0,0],[0,1.,0]],dtype=torch.float64)}
    a,b=Flow(),Flow()
    aa,ab=agent(),agent()
    original_a,original_b=a._sample_step,b._sample_step
    first=midpoint_step(a,original_a,state,.4,.2,aa,{})[0]
    second=rk2_step(b,original_b,state,.4,.2,ab,{},stage_fraction=.5)[0]
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    torch.testing.assert_close(aa['_init_diffusion_type_state'],ab['_init_diffusion_type_state'],rtol=0,atol=0)
    errors=[]
    for n in (6,12,24):
        f=Flow();ag=agent();current=state.clone()
        rng=torch.get_rng_state().clone()
        install_ralston(f)
        assert torch.equal(rng,torch.get_rng_state())
        for i in range(n):
            current=f._sample_step(current,1-i/n,1-(i+1)/n,ag,{})[0]
        assert f.calls==2*n
        torch.testing.assert_close(current[0],state[0],rtol=0,atol=0)
        torch.testing.assert_close(current[:,2:4].square().sum(-1),torch.ones(2,dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close(ag['_init_diffusion_type_state'][0],ag['_init_diffusion_type_source'][0],rtol=0,atol=0)
        torch.testing.assert_close(ag['_init_diffusion_type_state'][1],torch.tensor([-.3,1.2,-.1],dtype=torch.float64))
        errors.append(abs(float(current[1,0])-.4/1.4))
        clone=copy.deepcopy(f)
        assert clone._sample_step.__self__ is clone
    ratios=[errors[i]/errors[i+1] for i in range(2)]
    assert all(3.5<x<4.5 for x in ratios),ratios
    f=Flow();f.t_eps=.05;ag=agent()
    try:
        rk2_step(f,f._sample_step,state,.1,0.,ag,{})
    except ValueError:pass
    else:raise AssertionError('Below-floor stage must be rejected')
    rk2_step(f,f._sample_step,state,.15,0.,ag,{})
    result=dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        midpoint_special_case_bit_equal=True,call_counts=[12,24,48],
        nonlinear_ode_global_errors=errors,convergence_ratios=ratios,
        ego_and_type_preserved=True,circular_unit_norm=True,deepcopy_bound_to_copy=True,
        unchanged_torch_rng=True,velocity_floor_guard=True,
        limitation='Numerical/API proof only. No model quality gain established.')
    path=Path(__file__).resolve().parents[1]/'output/reports/init_diffusion_continuous_20261014/ralston_sampling_proof_20261015.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
