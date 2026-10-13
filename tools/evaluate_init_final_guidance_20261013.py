"""Paired full50K final-step-only guidance, avoiding attribute feedback."""
import argparse
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import evaluate_init_n20_evening_20261013 as sampler
read,save,sha=sampler.read,sampler.save,sampler.sha
BASE=sampler.ART.parent
ART=BASE/'guidance_final_only50000'
RUNS=ROOT/'logs/init_diffusion_evening_20261013/guidance_final_only50000'
OPTION='guide_final_only'
sampler.OPTIONS[OPTION]=dict(guidance=dict(scale=.5,t_max=.05,iters=5,margin=.1,vehicles_only=True),time_grid={})
STOP=False
original_grid_hook=sampler.make_grid_hook


def worker(path):
    def counted_hook(options,time_grid,**kwargs):
        base=original_grid_hook(options,time_grid,**kwargs)
        class Hook(type(base)):
            def on_test_start(self,trainer,model):
                super().on_test_start(trainer,model)
                flow=model.encoder.init_decoder.G1
                self.original_guided=flow._collision_guided_x0
                self.had_guided_attribute='_collision_guided_x0' in flow.__dict__
                self.record['total_guidance_calls']=0
                def counted(*args,**kw):
                    self.record['total_guidance_calls']+=1
                    return self.original_guided(*args,**kw)
                flow._collision_guided_x0=counted

            def on_test_end(self,trainer,model):
                assert self.record['total_guidance_calls']==self.record['sample_calls']
                self.record['observed_guidance_calls_per_sample']=1
                flow=model.encoder.init_decoder.G1
                if self.had_guided_attribute:flow._collision_guided_x0=self.original_guided
                else:del flow._collision_guided_x0
                super().on_test_end(trainer,model)
        return Hook(**kwargs)
    sampler.make_grid_hook=counted_hook
    sampler.worker(path)


def prepare():
    if (ART/'protocol.json').exists():return
    save(ART/'protocol.json',dict(registered_local=__import__('datetime').datetime.now().astimezone().isoformat(),
                                 options={OPTION:sampler.OPTIONS[OPTION]},scenes=50000,TRAIN=[4817],GEN=[2817,3817],
                                 checkpoint_updates=16384,NFE=20,parameters=8083978,
                                 fit_executed=False,objective_changed=False,
                                 purpose='One final uniform t=.05 model query; guidance has no later denoiser feedback into attributes. Position-dependent evaluator selection may still change metrics.',
                                 decision='Compare to already verified same-checkpoint full50K off anchor and earlier multi-step guides; no claim about128epoch weights.',
                                 sources={**sampler.source_hashes(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}))
    save(ART/'status.json',dict(status='WAITING_FOR_FULL_GUIDANCE_COMPLETE'))


def execute():
    global STOP
    def halt(*unused):
        global STOP
        STOP=True;sampler.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    plan=read(ART/'protocol.json')
    while not STOP and time.time()<sampler.END:
        status=read(BASE/'guidance_full50000/status.json')
        if status['status']=='COMPLETE':break
        if status['status'] not in ('WAITING_FOR_N20_COMPLETE','RUNNING_NATIVE_TEST'):
            save(ART/'status.json',dict(status='DEPENDENCY_INCOMPLETE',dependency=status));return
        time.sleep(10)
    if STOP or time.time()>=sampler.END:return
    assert plan['sources']=={**sampler.source_hashes(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    origins=read(BASE/'n20_sampler/frozen_endpoints.json');origin=origins['freshK8']
    rows=[r for r in read(BASE/'guidance_full50000/observations.json') if r['option']=='off_uniform']
    assert len(rows)==2
    for r in rows:
        proof=read(Path(r['folder'])/'verified_completion.json')
        for name,value in proof['artifact_sha256'].items():assert sha(Path(r['folder'])/name)==value
    save(ART/'imported_anchors.json',rows)
    files=sorted(p.name for p in (ROOT/'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    assert len(files)==50000
    sampler.ART,sampler.RUNS=ART,RUNS
    sampler.__file__=__file__  # Spawn this audited worker for guidance-call count.
    for gen in (2817,3817):
        cell=sampler.cell('freshK8',origin,gen,OPTION,files,'full50000')
        cell['sources'].update({str(Path(__file__).relative_to(ROOT)):sha(__file__)})
        if STOP or not sampler.run_cell(cell,rows):return
    comparisons=sampler.compare(rows,'freshK8',50000)
    for c in comparisons:
        c['optional_noninferior_guidance']=(c['option']!= 'off_uniform' and c['R_focus']<=1.01
                                            and all(c['ratios'][k]<=1.05 for k in sampler.JSD)
                                            and c['onroad_delta']>=-.02 and c['ratios']['collision_rate']<=.5)
    save(ART/'comparisons.json',comparisons)
    save(ART/'status.json',dict(status='COMPLETE',completed_cells=len(rows),actual_full50000=True,full128epoch_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--cell',type=Path);args=p.parse_args()
    if args.cell:worker(args.cell)
    else:execute() if args.execute else prepare()
