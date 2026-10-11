# InitDiffusion loss search — 2026-10-11

Compact results retained with the repository documentation. See the [report](../../init_diffusion_loss_search_20261011.md) for protocol, recipe, metric definitions, interpretation and limitations.

- `training_runs.csv`: all94 completed training runs;128-step search and512-step confirmation have different budgets/scopes and must not be ranked together without pairing.
- `search_results.csv`: four-round primary search results including warm controls.
- `confirmation_summary.csv` and `paired_observations.csv`: repeated-seed confirmations;generation seeds are nested within training seeds;matched/full-reference scopes reuse the same generated scenes.
- `full50k_results.csv`: the locked median-seed recipe and warm baseline, each generating50000 scenes.
- `selected_recipe.json`, `selected_loss_recipe.json`, `search_plan.json`: declared selection rule,resolved loss and adaptive decisions.
- Audits, histogram plots/statistics and gradient measurements preserve verification evidence.

Full checkpoints,manifests,training traces,per-run configurations andconsole logs remain under `logs/component_loss_quick/` in the originalworkspace. The selected checkpoint is `logs/component_loss_quick/confirmation_seed2817/angle_mse_128/last.ckpt`. Large weights are not duplicated here. JSON absolute paths identify the original local run;the tables and plots remain readable without those weights.
