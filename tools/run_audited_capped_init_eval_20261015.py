"""Native evaluation with source/NFE audits and a process-local memory cap."""
from pathlib import Path
import argparse,functools,gc,hashlib,json,runpy,sys,time
from types import MethodType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MODEL=None


def count_forward(module,inputs):module._eval_audit_calls+=1

def audited_guidance(self,*args,**kwargs):
    self._eval_audit_guidance_calls+=1
    return self._eval_audit_original_guidance(self,*args,**kwargs)

def audited_sample(self,*args,**kwargs):
    before=self.model._eval_audit_calls;guidance_before=self._eval_audit_guidance_calls
    result=self._eval_audit_original_sample(self,*args,**kwargs)
    agent=kwargs.get('tokenized_agent',args[0] if args else None)
    h=hashlib.sha256()
    for key in ('gen_noise','_init_diffusion_type_source'):
        value=agent[key].detach().cpu().contiguous()
        h.update(str((key,str(value.dtype),tuple(value.shape))).encode());h.update(value.numpy().tobytes())
    self._eval_audit_rows.append(dict(source_sha256=h.hexdigest(),NFE=self.model._eval_audit_calls-before,
        guidance_calls=self._eval_audit_guidance_calls-guidance_before,agents=len(agent['batch'])))
    return result


def main():
    global MODEL
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--gpu-memory-fraction',type=float,required=True);p.add_argument('--resource-report',type=Path,required=True)
    p.add_argument('--audit-report',type=Path,required=True);p.add_argument('--histogram-report',type=Path,required=True)
    a,remaining=p.parse_known_args();assert 0<a.gpu_memory_fraction<=.5
    for path in (a.resource_report,a.audit_report,a.histogram_report):path.parent.mkdir(parents=True,exist_ok=True)
    gc.disable();gc.collect=lambda *args,**kwargs:0
    import jinja2,torch,numpy as np
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,0)
    from src.smart.model.smart_gail import SMART_GAIL
    original_init=SMART_GAIL.__init__
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        global MODEL
        original_init(self,*args,**kwargs);MODEL=self
        before=torch.get_rng_state().clone();flow=self.encoder.init_decoder.G1
        flow.model._eval_audit_calls=0;flow.model.register_forward_pre_hook(count_forward)
        flow._eval_audit_rows=[];flow._eval_audit_guidance_calls=0
        flow._eval_audit_original_sample=flow.sample.__func__;flow.sample=MethodType(audited_sample,flow)
        flow._eval_audit_original_guidance=flow._collision_guided_x0.__func__
        flow._collision_guided_x0=MethodType(audited_guidance,flow)
        assert torch.equal(before,torch.get_rng_state())
    SMART_GAIL.__init__=init;success=False;begin=time.monotonic()
    try:
        sys.argv=[str(ROOT/'src/run.py'),*remaining];runpy.run_path(str(ROOT/'src/run.py'),run_name='__main__')
        from src.smart.metrics.metric_core import SPECS,finalize_metrics
        evaluator=MODEL.sd_evaluator;flow=MODEL.encoder.init_decoder.G1
        assert flow._eval_audit_rows and set(r['NFE'] for r in flow._eval_audit_rows)=={24}
        hist={name:dict(edges=np.arange(lo,hi+step,step).tolist(),generated=evaluator.generated.histograms[i].tolist(),
                       matched_gt=evaluator.real.histograms[i].tolist(),multiplier=multiplier)
              for i,(name,lo,hi,step,multiplier) in enumerate(SPECS)}
        a.histogram_report.write_text(json.dumps(hist,indent=2)+'\n')
        a.audit_report.write_text(json.dumps(dict(status='PASS',rows=flow._eval_audit_rows,
            recomputed_metrics=finalize_metrics(evaluator.generated,evaluator.real),
            decoder_ema_updates=MODEL.encoder.init_decoder.ema.num_updates,hook_install_CPU_RNG_unchanged=True),indent=2)+'\n')
        success=True
    finally:
        SMART_GAIL.__init__=original_init
        a.resource_report.write_text(json.dumps(dict(native_cli_success=success,process_memory_fraction=a.gpu_memory_fraction,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            elapsed_seconds=time.monotonic()-begin,command=remaining),indent=2)+'\n')

if __name__=='__main__':main()
