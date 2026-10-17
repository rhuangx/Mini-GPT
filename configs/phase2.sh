#!/usr/bin/env bash
# Phase 2 runs on configs/baseline.json; `python -m mini_gpt.sweep` prints each budget's lines. Rerunning
# skips finished runs and resumes interrupted ones.
set -euo pipefail
cd "$(dirname "$0")/.."
train() { .venv/bin/python -m mini_gpt.train configs/baseline.json "$@"; }

# Stage 0: two LR checks, each on a planned run, so its planned LR is also the sweep's point.
# Long horizon: 3.1M at 3e16, 9.6x its compute-optimal tokens. Planned 0.00151 is LR*(N)
# lowered by 9.6^-0.25; 0.00266 is LR*(N) uncorrected; 0.000755 brackets below.
train --set model.d_model=256 model.n_layers=4 train.flops=3e+16 train.lr=0.00151 train.micro_batch=8
train --set model.d_model=256 model.n_layers=4 train.flops=3e+16 train.lr=0.00266 train.micro_batch=8
train --set model.d_model=256 model.n_layers=4 train.flops=3e+16 train.lr=0.000755 train.micro_batch=8
# Large size: 85M at 1e17, 3.4x past the largest size Phase 1 swept, at 0.09x its tokens as the
# sweep uses it. Planned 0.000287, then 2x and 0.5x.
train --set model.d_model=768 model.n_layers=12 train.flops=1e+17 train.lr=0.000287 train.micro_batch=8
train --set model.d_model=768 model.n_layers=12 train.flops=1e+17 train.lr=0.000574 train.micro_batch=8
train --set model.d_model=768 model.n_layers=12 train.flops=1e+17 train.lr=0.000143 train.micro_batch=8
