#!/bin/bash

set -euo pipefail

# RAL experiment evaluation entry.
# Important:
# 1. This script only evaluates existing checkpoints.
# 2. It does NOT train any method.
# 3. Fair protocol: train each method first with `bash run.sh`, then run this script.

CONFIG=${CONFIG:-configs/paper_final_full.args}
OURS_CKPT=${OURS_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-20-38-15/checkpoint0014.pth}
FIXED_CKPT=${FIXED_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-21-41-35/checkpoint0014.pth}
NONDIFF_CKPT=${NONDIFF_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-22-57-27/checkpoint0014.pth}
BLIND_CKPT=${BLIND_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-18-00-42/checkpoint0015.pth}

EPISODES=${EPISODES:-20}
PLOT_LEVEL=${PLOT_LEVEL:-l3}
OUTPUT_DIR=${OUTPUT_DIR:-}
INCLUDE_BASE=${INCLUDE_BASE:-0}

if [ ! -f "$CONFIG" ]; then
	echo "[error] config not found: $CONFIG"
	exit 1
fi

for ckpt in "$OURS_CKPT" "$FIXED_CKPT" "$NONDIFF_CKPT"; do
	if [ ! -f "$ckpt" ]; then
		echo "[error] checkpoint not found: $ckpt"
		exit 1
	fi
done

cmd=(
	python3 paper/experiment/run_ral_eval_suite.py
	--config "$CONFIG"
	--ours_ckpt "$OURS_CKPT"
	--fixed_ckpt "$FIXED_CKPT"
	--nondiff_ckpt "$NONDIFF_CKPT"
	--episodes_per_condition "$EPISODES"
	--plot_level "$PLOT_LEVEL"
)

if [ "$INCLUDE_BASE" = "1" ]; then
	cmd+=(--include_base)
fi

if [ -n "$OUTPUT_DIR" ]; then
	cmd+=(--output_dir "$OUTPUT_DIR")
fi

if [ -n "$BLIND_CKPT" ]; then
	if [ ! -f "$BLIND_CKPT" ]; then
		echo "[error] blind checkpoint not found: $BLIND_CKPT"
		exit 1
	fi
	cmd+=(--include_blind --blind_ckpt "$BLIND_CKPT")
fi

echo "[eval-suite] evaluate only; no training will be run."
echo "[eval-suite] config        : $CONFIG"
echo "[eval-suite] ours_ckpt     : $OURS_CKPT"
echo "[eval-suite] fixed_ckpt    : $FIXED_CKPT"
echo "[eval-suite] nondiff_ckpt  : $NONDIFF_CKPT"
if [ -n "$BLIND_CKPT" ]; then
	echo "[eval-suite] blind_ckpt    : $BLIND_CKPT"
else
	echo "[eval-suite] blind_ckpt    : <disabled>"
fi
echo "[eval-suite] episodes/cond : $EPISODES"
echo "[eval-suite] plot_level    : $PLOT_LEVEL"

"${cmd[@]}"
