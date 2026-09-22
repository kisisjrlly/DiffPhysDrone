#!/bin/bash
set -euo pipefail

ckpt=${CKPT:-}
config=${CONFIG:-configs/gray_gate_fixed.args}
eval_episodes=${EVAL_EPISODES:-10}
vis_enable=${VIS_ENABLE:-0}
extra=${EVAL_EXTRA_ARGS:-}
trace_csv=${EVAL_TRACE_CSV:-}
episode_csv=${EVAL_EPISODE_CSV:-}
log_to_file=${LOG_TO_FILE:-0}

if [ -z "$ckpt" ]; then
  echo "[error] set CKPT=checkpoint/.../checkpointXXXX.pth" >&2
  exit 1
fi
if [ ! -f "$ckpt" ]; then
  echo "[error] checkpoint not found: $ckpt" >&2
  exit 1
fi
if [ ! -f "$config" ]; then
  echo "[error] config not found: $config" >&2
  exit 1
fi

if [ -n "${PYTHON_BIN:-}" ]; then
  py_bin="$PYTHON_BIN"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  py_bin="${CONDA_PREFIX}/bin/python"
elif [ -x "${HOME}/miniconda3/envs/mappo-mpc/bin/python" ]; then
  py_bin="${HOME}/miniconda3/envs/mappo-mpc/bin/python"
else
  py_bin=python
fi

cfg_args=$(sed -E 's/[[:space:]]*#.*$//' "$config" | grep -Ev '^[[:space:]]*$' | xargs)
vis_args=""
[ "$vis_enable" = "1" ] && vis_args="--vis_enable"

csv_args=""
[ -n "$trace_csv" ] && csv_args="$csv_args --eval_trace_csv $trace_csv"
[ -n "$episode_csv" ] && csv_args="$csv_args --eval_episode_csv $episode_csv"

cmd="$py_bin -u eval.py $cfg_args $extra --resume $ckpt --wandb_disabled --eval_episodes $eval_episodes $vis_args $csv_args"
echo "config     : $config"
echo "checkpoint : $ckpt"
echo "command    : $cmd"

if [ "$log_to_file" = "1" ]; then
  mkdir -p logs
  logfile="logs/eval-gray-$(date +%Y-%m-%d-%H-%M-%S).log"
  eval "$cmd" > "$logfile" 2>&1
  echo "log: $logfile"
else
  eval "$cmd"
fi
