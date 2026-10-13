"""Plot existing paired histogram counts; never replace mean-GEN JSD scoring."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import METRICS, read, save, sha

ART = ROOT / "output/reports/wandb_cc_20261013"
FOLDERS = {
    "657": ROOT / "logs/init_diffusion_657_edges_20261013/medium_control657/control657/eval_step_4096",
    "K8": ROOT / "logs/init_diffusion_657_edges_20261013/medium_a2a_k8/a2a_k8/eval_step_4096",
    "Lane + agent_max0.5": ROOT / "logs/init_diffusion_657_compact_20261013/medium_lane_agentmax05/lane_agentmax05/eval_step_4096",
}


def main():
    diagnostics, pooled, source = {}, {}, {}
    for label, folder in FOLDERS.items():
        groups = [folder / "ema", folder / "eval_seed_3817/ema"]
        records = [read(p / "feature_histograms.json") for p in groups]
        pooled[label] = {}
        diagnostics[label] = {}
        for i, key in enumerate(METRICS[:6]):
            data = records[0][key]
            lo, step = data["minimum"], data["bin_width"]
            generated = sum(np.asarray(r[key]["generated"]) for r in records)
            target = sum(np.asarray(r[key]["matched_gt"]) for r in records)
            midpoints = lo + (np.arange(len(generated)) + .5) * step
            p, q = generated / generated.sum(), target / target.sum()
            mean, std = float(p @ midpoints), float(np.sqrt(p @ (midpoints-p@midpoints)**2))
            diagnostics[label][key] = {"approx_binned_mean": mean, "approx_binned_std": std,
                "first_bin_probability": float(p[0]), "last_bin_probability": float(p[-1]),
                "gt_mean": float(q @ midpoints), "gt_std": float(np.sqrt(q @ (midpoints-q@midpoints)**2)),
                "generated_count": int(generated.sum()), "gt_count": int(target.sum()),
                "pooled_histogram_jsd_for_plot_only": float(jensenshannon(p,q)**2 * (10 if i < 2 else 100))}
            pooled[label][key] = (midpoints, p, q, step)
            for group, record in zip(groups, records):
                native = read(group / "sd_agent_metrics.json")
                metric = float(jensenshannon(record[key]["generated"], record[key]["matched_gt"])**2 * (10 if i < 2 else 100))
                assert abs(metric-native["agent_metrics"][key]) < 1e-10, (group,key)
        for group in groups:
            for name in ("feature_histograms.json", "sd_agent_metrics.json"):
                source[str((group/name).relative_to(ROOT))] = sha(group/name)
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
    limits = {"nearest_dist_jsd": (0,30), "ang_dev_jsd":(-90,90), "length_jsd":(2,9), "width_jsd":(1,3.2), "speed_jsd":(0,25)}
    for ax, key in zip(axes.flat,METRICS[:6]):
        x,p,q,step=pooled["657"][key]
        ax.step(x,q,where="mid",label="Matched GT",color="#172338",linewidth=2)
        for label in FOLDERS:
            x,p,q,step=pooled[label][key]
            ax.step(x,p,where="mid",label=label,alpha=.85)
        ax.set_title(key)
        ax.set_ylabel("Probability per native bin")
        if key in limits:
            ax.set_xlim(*limits[key])
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Historical step4096 / TRAIN4817 / 4096 scenes / two GEN histogram counts pooled\nDiagnostic distributions, not a replacement for mean-GEN JSD scoring",fontsize=13)
    fig.savefig(ART / "historical_jsd_histograms.png",dpi=160)
    plt.close(fig)
    save(ART / "historical_histogram_diagnostics.json", {"source_sha256": source,"diagnostics": diagnostics,
         "native_metric_reconstruction_checks": 36, "all_checks_pass": True,
         "binned_moments_are_approximate_and_clipped": True,
         "pooled_histogram_jsd_not_used_in_rankings": True})
    print(json.dumps({"image": str(ART / "historical_jsd_histograms.png"), "native_checks_pass":36}))


if __name__ == "__main__":
    main()
