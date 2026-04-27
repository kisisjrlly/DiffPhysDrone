#!/bin/bash

set -euo pipefail

# RAL experiment evaluation entry.
# Important:
# 1. This script only evaluates existing checkpoints.
# 2. It does NOT train any method.
# 3. Fair protocol: train each method first with `bash run.sh`, then run this script.

CONFIG=${CONFIG:-configs/paper_final_full.args}
OURS_CKPT=${OURS_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-25-23-45-54/checkpoint0049.pth}
FIXED_CKPT=${FIXED_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-26-11-24-40/checkpoint0034.pth}
FIXED_RANDOM_CKPT=${FIXED_RANDOM_CKPT:-}
NONDIFF_CKPT=${NONDIFF_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-25-20-16-12/checkpoint0049.pth}
BLIND_CKPT=${BLIND_CKPT:-/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-25-16-19-31/checkpoint0049.pth}

EPISODES=${EPISODES:-20}
PLOT_LEVEL=${PLOT_LEVEL:-l3}
PLOT_REGIME=${PLOT_REGIME:-glare}
SLOTS=${SLOTS:-"far_left left right far_right"}
REGIMES=${REGIMES:-"glare specular dark"}
INCLUDE_OURS_ZERO=${INCLUDE_OURS_ZERO:-1}
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
	"${PYTHON_BIN:-python3}" paper/experiment/run_ral_eval_suite.py
	--config "$CONFIG"
	--ours_ckpt "$OURS_CKPT"
	--fixed_ckpt "$FIXED_CKPT"
	--nondiff_ckpt "$NONDIFF_CKPT"
		--episodes_per_condition "$EPISODES"
		--plot_level "$PLOT_LEVEL"
		--plot_regime "$PLOT_REGIME"
		--slots $SLOTS
		--regimes $REGIMES
	)

if [ -n "$FIXED_RANDOM_CKPT" ]; then
	if [ ! -f "$FIXED_RANDOM_CKPT" ]; then
		echo "[error] fixed-random checkpoint not found: $FIXED_RANDOM_CKPT"
		exit 1
	fi
	cmd+=(--include_fixed_random --fixed_random_ckpt "$FIXED_RANDOM_CKPT")
fi

if [ "$INCLUDE_OURS_ZERO" = "1" ]; then
	cmd+=(--include_ours_zero_ablation)
fi

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
if [ -n "$FIXED_RANDOM_CKPT" ]; then
	echo "[eval-suite] randfix_ckpt  : $FIXED_RANDOM_CKPT"
else
	echo "[eval-suite] randfix_ckpt  : <disabled>"
fi
echo "[eval-suite] nondiff_ckpt  : $NONDIFF_CKPT"
if [ -n "$BLIND_CKPT" ]; then
	echo "[eval-suite] blind_ckpt    : $BLIND_CKPT"
else
	echo "[eval-suite] blind_ckpt    : <disabled>"
fi
echo "[eval-suite] episodes/cond : $EPISODES"
echo "[eval-suite] plot_level    : $PLOT_LEVEL"
echo "[eval-suite] plot_regime   : $PLOT_REGIME"
echo "[eval-suite] slots         : $SLOTS"
echo "[eval-suite] regimes       : $REGIMES"
echo "[eval-suite] ours_zero     : $INCLUDE_OURS_ZERO"

"${cmd[@]}"
