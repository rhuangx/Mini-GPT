#!/usr/bin/env bash
# Phase 1 runs on configs/baseline.json. Rerunning skips finished runs and resumes interrupted ones.
set -euo pipefail
cd "$(dirname "$0")/.."
train() { .venv/bin/python -m mini_gpt.train configs/baseline.json "$@"; }

# Sizes, each at D = 20N tokens, its compute-optimal budget:
#   d256 L4   3.1M non-embedding params  3.12e15 FLOPs
#   d384 L6  10.6M                       2.46e16
#   d512 L8  25.3M                       1.15e17
S="model.d_model=256 model.n_layers=4"

# LR sweep, smallest size.
for lr in 5e-4 1e-3 2e-3 4e-3 8e-3; do train --set $S train.flops=3.12e15 train.lr=$lr; done

# LR vs horizon: the same grid at 1/4x and 4x the compute-optimal tokens.
for flops in 7.81e14 1.25e16; do
  for lr in 5e-4 1e-3 2e-3 4e-3 8e-3; do train --set $S train.flops=$flops train.lr=$lr; done
done

# LR sweep, mid size. Grid 2x below the smallest's: there 1e-3 to 4e-3 tie and 8e-3 breaks down.
M="model.d_model=384 model.n_layers=6"
for lr in 2.5e-4 5e-4 1e-3 2e-3 4e-3; do train --set $M train.flops=2.46e16 train.lr=$lr; done

# Noise floor, smallest size: 3 seeds at 1e16 FLOPs at its LR*.
for seed in 0 1 2; do train --set $S train.flops=1e16 train.lr=2e-3 train.seed=$seed; done

# Noise floor, mid size: 3 seeds at 1e16 FLOPs at its LR*.
for seed in 0 1 2; do train --set $M train.flops=1e16 train.lr=1e-3 train.seed=$seed; done

# LR sweep, largest size, centred on LR*(N) from the two smaller sizes (~6e-4). Their cliffs, 8e-3 at
# 3.1M and 4e-3 at 10.6M, put this one's at 2e-3 or below. Most informative LRs first.
L="model.d_model=512 model.n_layers=8"
# 1.25e-4 skipped: 2.5e-4 already finished more than 2x the noise floor above the best.
for lr in 5e-4 1e-3 2.5e-4 2e-3; do train --set $L train.flops=1.15e17 train.lr=$lr; done
