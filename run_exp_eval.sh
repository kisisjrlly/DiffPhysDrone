#!/usr/bin/env bash
set -euo pipefail

# Current final-evaluation entrypoint. It extracts checkpoints from training logs,
# runs the checkpoint eval suite, and regenerates journal figures/tables.

PYTHON_BIN=${PYTHON_BIN:-/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python}
CONFIG=${CONFIG:-configs/slit_active_sensing.args}
OUT_DIR=${OUT_DIR:-paper/experiment/results/final_semantics_v3_eval_20260508}
EPISODES_PER_SCENE=${EPISODES_PER_SCENE:-100}
SCENARIOS=${SCENARIOS:-"glare specular dark"}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}

LOGS=("$@")
if [ "${#LOGS[@]}" -eq 0 ]; then
	LOGS=(
		logs/2026-05-08-00-44-21-slit_active_sensing_auto_fix-cam-fixed_grad-detached_depth-depth.log
		logs/2026-05-08-01-17-20-slit_active_sensing_auto_randfix-cam-fixed_random_static_grad-detached_depth-depth.log
		logs/2026-05-08-01-49-58-slit_active_sensing_auto_nondiff-cam-learned_grad-detached_depth-depth.log
		logs/2026-05-08-02-31-45-slit_active_sensing_auto_zero-cam-fixed_grad-detached_depth-zero.log
		logs/2026-05-08-03-04-26-slit_active_sensing_auto_flightonly-cam-learned_grad-detached_depth-depth.log
	)
fi

cmd=(
	"$PYTHON_BIN" -u tools/run_final_eval_from_logs.py
	"${LOGS[@]}"
	--config "$CONFIG"
	--out_dir "$OUT_DIR"
	--episodes_per_scene "$EPISODES_PER_SCENE"
	--scenarios $SCENARIOS
	--seed "$SEED"
	--device "$DEVICE"
	--python "$PYTHON_BIN"
)

if [ "${SKIP_EVAL:-0}" = "1" ]; then
	cmd+=(--skip_eval)
fi
if [ "${SKIP_ASSETS:-0}" = "1" ]; then
	cmd+=(--skip_assets)
fi
if [ "${SKIP_DEPTH_SEQUENCES:-0}" = "1" ]; then
	cmd+=(--skip_depth_sequences)
fi

echo "[run-exp-eval] ${cmd[*]}"
"${cmd[@]}"
