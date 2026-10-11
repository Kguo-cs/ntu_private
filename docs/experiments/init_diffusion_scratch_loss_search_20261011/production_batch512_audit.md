# Production batch512 transfer audit

This document proposes a fresh-training confirmation; it does not report a GPU run.

The existing base experiment is batch512/LR5e-4 with a pretrained checkpoint and frozen map. The genuine scratch config and search use batch64/LR1e-4 and random trainable map + G1. Changing all of these together does not isolate loss effects.

## Heldout raw-scene evidence

The 4,096 heldout filenames represent 4,096 distinct raw Waymo keys (TFRecord + scene index), with zero raw-key or filename overlap with the 1,024 search cache files (1,023 unique raw keys). All selected lg_type values are 0. The audit JSON contains the first 512 such heldout filenames and can be passed directly to --eval-manifest.

## Equal examples, different update counts

| Batch64 step | Batch512 step | Seen scenes | EMA mean age in scenes:64 | EMA mean age in scenes:512 |
|---:|---:|---:|---:|---:|
| 2048 | 256 | 131072 | 13113.6 | 13158.4 |
| 4096 | 512 | 262144 | 26220.8 | 26265.6 |
| 6144 | 768 | 393216 | 39328.0 | 39372.8 |

torch_ema currently uses its update-count warmup, d=min(.9999,(1+u)/(10+u)). Thus the present equal-example EMA mean ages are almost identical, despite different optimizer-step horizons. Once the configured .9999 cap dominates (about 90,000 updates), its 10,000-update horizon corresponds to 640,000 scenes at batch64 and 5,120,000 scenes at batch512. EMA here covers G1; the trainable shared map remains online.

## Fast confirmation command

The four listed controls/candidates may be reduced to the final two contenders plus legacy once the 6,144-stage selection is complete. Use fresh seed817; do not load or resume batch64 weights.

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/scratch_component_loss_sweep.py \
  --variant-file logs/component_loss_scratch/design/adaptive_variants.json \
  --variants legacy,geometry_strong,geometry_strong_heading_l1,geo_mse32_col20 \
  --recipe logs/component_loss_scratch/design/anchor_recipe.json \
  --normalizer-json docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json \
  --batch-size 512 --steps 768 --checkpoint-steps 256,512 \
  --scenes 512 --eval-batch-size 128 \
  --eval-manifest docs/experiments/init_diffusion_scratch_loss_search_20261011/production_batch512_audit.json \
  --seed 817 --eval-seed 2817 --workers 0 --lr 0.0001 --evaluate-ema \
  --output logs/component_loss_scratch/production512_confirm817
```

This sees 393,216 training scenes at the final budget, matching 6,144 batch64 steps and staying inside one real data epoch. The 256/512 checkpoints match 2,048/4,096 batch64 steps. Keep LR1e-4 first to isolate batch transfer; LR5e-4 is an additional experiment. Batch512 memory capacity must be checked when the job starts.

Use online for primary ranking and show G1 EMA separately. Check the same heldout split, on-road coverage and generated vehicle counts. A 512-scene one-seed confirmation is a screening result; final contenders should also be checked on the full 4,096 heldout set and another initialization seed.

## Fresh 2,048-step extension

The audited --allow-multiple-epochs option requires fresh weights and zero train offset. It repeats the seeded full manifest; native training instead shuffles each epoch. With 493,542 scenes, batch512 and drop_last=False, each full epoch contains 964 batches (last batch 486 scenes). At 2,048 updates the batch-end checkpoint should be epoch 2 with scheduler last_epoch 2. Actual examples are 1,048,524, which is 52 fewer than 2,048×512. Resume-map remains unsupported for this multi-epoch mode.

The installed torch_ema defaults use_num_updates=True. At 512×768 and 64×6144, effective decay is 0.9884319 and 0.9985375 respectively; mean EMA age in seen scenes is 39,372.8 and 39,328.0. At 512×2048, effective decay is 0.9956268 and mean age is104893.72 seen scenes, accounting for partial final batches. EMA covers G1 only; the shared map stays online. See fresh_batch512_multi_epoch_audit.json for exact values.

The original cache suffix was independently checked in 64 official .pkl files selected from search/heldout: saved scene_timestep equals the final suffix in all 64, and none has a map_id field. This rules out accidentally treating a filename map category as a timestep for those checks. See filename_suffix_evidence.json.
