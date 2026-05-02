#!/bin/bash
set -euo pipefail

# Minimal eval wrapper. Usage:
#   CKPT=checkpoint/.../checkpointXXXX.pth bash eval.sh
# Optional:
#   EVAL_EXTRA_ARGS="--scenarios glare --sun_glare_eval_slot right" bash eval.sh

export all_proxy=${all_proxy:-http://127.0.0.1:7890}

task=${TASK:-paper_final_full}
ckpt_path=${CKPT:-checkpoint/2026-05-02-21-47-04/checkpoint0009.pth}
eval_episodes=${EVAL_EPISODES:-10}
vis_episode_idx=${VIS_EPISODE_IDX:--1}
eval_batch_size=${EVAL_BATCH_SIZE:-1}
vis_enable=${VIS_ENABLE:-1}
eval_extra_args=${EVAL_EXTRA_ARGS:-}
eval_trace_csv=${EVAL_TRACE_CSV:-}
eval_episode_csv=${EVAL_EPISODE_CSV:-}
log_to_file=${LOG_TO_FILE:-0}

if [ -z "$ckpt_path" ]; then
	echo "[error] set CKPT=checkpoint/.../checkpointXXXX.pth"
	exit 1
fi
if [ ! -f "$ckpt_path" ]; then
	echo "[error] checkpoint not found: $ckpt_path"
	exit 1
fi

if [ -n "${PYTHON_BIN:-}" ]; then
	py_bin="$PYTHON_BIN"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
	py_bin="${CONDA_PREFIX}/bin/python"
elif [ -x "${HOME}/miniconda3/envs/mappo-mpc/bin/python" ]; then
	py_bin="${HOME}/miniconda3/envs/mappo-mpc/bin/python"
elif command -v python >/dev/null 2>&1; then
	py_bin="python"
else
	py_bin="python3"
fi

cfg_file="configs/${task}.args"
if [ ! -f "$cfg_file" ]; then
	echo "[error] config file not found: $cfg_file"
	exit 1
fi

cfg_args=$(sed -E 's/[[:space:]]*#.*$//' "$cfg_file" | grep -Ev '^[[:space:]]*$' | xargs)
vis_args=""
if [ "$vis_enable" = "1" ]; then
	vis_args="--vis_enable"
elif [ "$vis_enable" != "0" ]; then
	echo "[error] invalid VIS_ENABLE=$vis_enable"
	exit 1
fi
csv_args=""
[ -n "$eval_trace_csv" ] && csv_args="$csv_args --eval_trace_csv $eval_trace_csv"
[ -n "$eval_episode_csv" ] && csv_args="$csv_args --eval_episode_csv $eval_episode_csv"

mkdir -p logs
log_file="logs/eval-${task}-$(date +%Y-%m-%d-%H-%M-%S).log"
cmd="$py_bin -u eval.py $cfg_args $eval_extra_args --resume $ckpt_path $vis_args --wandb_disabled --eval_episodes $eval_episodes --vis_episode_idx $vis_episode_idx --batch_size $eval_batch_size $csv_args"

echo "using config     : $cfg_file"
echo "using checkpoint : $ckpt_path"
echo "eval episodes    : $eval_episodes"
echo "eval extra args  : ${eval_extra_args:-<none>}"

if [ "$log_to_file" = "1" ]; then
	eval "$cmd" > "$log_file" 2>&1
	echo "log file: $log_file"
elif [ "$log_to_file" = "0" ]; then
	eval "$cmd"
else
	echo "[error] invalid LOG_TO_FILE=$log_to_file"
	exit 1
fi
