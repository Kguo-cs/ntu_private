"""CPU-only diagnostic of generated raw scalar speeds before physical clipping.

Uses existing saved models and native evaluation without modifying predictions.
This small CPU cohort is diagnostic, never part of GPU candidate ranking.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import hydra
import lightning as L
from omegaconf import OmegaConf,open_dict
import torch
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
from src.smart.diffusion.denoiser import InitDenoiser

ART=ROOT/"output/reports/init_diffusion_jsd_round2_20261013/cpu_speed_probe"


def main():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("medium")
    ART.mkdir(parents=True,exist_ok=True)
    names=read(ROOT/"tmp/control657_paired_20261012/eval4096_files.json")[:256]
    manifest=ART/"eval_files.pkl"
    with manifest.open("wb") as stream:pickle.dump({"files":names},stream)
    oldrows=read(ROOT/"output/reports/wandb_cc_20261013/jsd_priority/observations.json")
    original=InitDenoiser.get_output
    original_input=InitDenoiser.get_input
    captured=[]
    truths=[]
    def observe_input(self,agents,*args,**kwargs):
        result=original_input(self,agents,*args,**kwargs)
        truths.append(result[0].detach()[agents["type"]==0,6].cpu())
        return result
    def observe(self,prediction,agents):
        labels=agents.get("_init_diffusion_generated_type",agents["type"])
        mask=labels==0
        captured.append(prediction.detach()[mask,6].cpu())
        return original(self,prediction,agents)
    InitDenoiser.get_output=observe
    InitDenoiser.get_input=observe_input
    results=[]
    for arm in ("fresh657","lane_agentmax05"):
        row=next(r for r in oldrows if r["variant"]==arm and r["seed"]==5817 and r["budget"]==4096)
        cfg=OmegaConf.load(Path(row["checkpoint"]).parent/"config.yaml")
        folder=ART/arm;folder.mkdir(exist_ok=True)
        with open_dict(cfg):
            cfg.model.model_config.initial_map_checkpoint=None
            cfg.model.model_config.decoder.init_diffusion.normalizer_path=None
            cfg.model.model_config.sd_r_reference=None
            # expected_scenes validates the backing reference store, not the
            # generated subset selected by the datamodule's sample manifest.
            cfg.model.model_config.sd_expected_scenes=50000
            cfg.model.model_config.sd_require_full_set=False
            cfg.data.scenario_dreamer_eval_set=str(manifest)
            cfg.data.test_batch_size=32
            cfg.data.num_workers=0
            cfg.data.pin_memory=False
            cfg.data.persistent_workers=False
        L.seed_everything(2817,workers=True)
        model=hydra.utils.instantiate(cfg.model,_recursive_=False)
        state=torch.load(row["checkpoint"],map_location="cpu",weights_only=False)
        model.load_state_dict(state["state_dict"],strict=True)
        del state
        model.encoder.init_decoder.use_ema=True
        model.video_dir=folder/"videos";model.video_dir.mkdir(exist_ok=True)
        dm=hydra.utils.instantiate(cfg.data,_recursive_=False)
        captured.clear()
        truths.clear()
        started=time.monotonic()
        trainer=L.Trainer(accelerator="cpu",devices=1,precision="32-true",logger=False,
            enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False,
            default_root_dir=str(folder))
        trainer.test(model,datamodule=dm,verbose=False)
        values=torch.cat(captured)
        truth=torch.cat(truths)
        report=read(folder/"sd_agent_metrics.json")
        assert report["num_samples"]==256 and len(values)==report["num_generated_vehicles"]
        assert len(truth)==report["num_gt_vehicles"]
        assert torch.isfinite(values).all()
        result={"arm":arm,"scenes":256,"GEN":2817,"TRAIN":5817,"budget":4096,
            "device":"cpu","not_gpu_quality_evidence":True,"checkpoint_sha256":sha(row["checkpoint"]),
            "generated_vehicles":len(values),"raw_negative_fraction":float((values<0).float().mean()),
            "raw_nonnegative_below1_fraction":float(((values>=0)&(values<1)).float().mean()),
            "physical_below1_fraction":float((values.clamp_min(0)<1).float().mean()),
            "raw_above15_fraction":float((values>=15).float().mean()),
            "GT_below1_fraction":float((truth<1).float().mean()),
            "GT_near_zero_fraction":float((truth<.01).float().mean()),
            "physical_near_zero_fraction":float((values.clamp_min(0)<.01).float().mean()),
            "GT_above15_fraction":float((truth>=15).float().mean()),
            "raw_speed_quantiles":dict(zip(["q01","q10","q50","q90","q99"],
                values.quantile(torch.tensor([.01,.1,.5,.9,.99])).tolist())),
            "seconds":time.monotonic()-started,"prediction_mutated":False}
        results.append(result);save(ART/"results.json",{"generated_utc":now(),"results":results})
        print(json.dumps(result),flush=True)
        del model,dm,trainer
    InitDenoiser.get_output=original
    InitDenoiser.get_input=original_input


if __name__=="__main__":main()
