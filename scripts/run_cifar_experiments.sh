#!/usr/bin/env bash
# Reproduces the CIFAR-10 experiments of Sec. 4.2: plain vs. residual nets of
# depth 20/32/44/56 (Fig. 6 left/middle) and ResNet-110 (Table 6).
# Two streams share the GPU: one for residual nets, one for plain nets.
set -e
PY=${PYTHON:-python}
SEED=${SEED:-0}

( for d in 20 32 44 56 110; do
    $PY train_cifar.py --depth $d --seed $SEED --resume
  done ) &
( for d in 20 32 44 56; do
    $PY train_cifar.py --depth $d --plain --seed $SEED --resume
  done ) &
wait
