#!/bin/bash
set -euo pipefail

# Grayscale/IMX900 training wrapper.
# Usage:
#   TASK=gray_gate_fixed bash run.sh
#   TASK=gray_camera_full RUN_EXTRA_ARGS="--num_iters 100" bash run.sh

task=${TASK:-gray_gate_fixed}
log_to_file=${LOG_TO_FILE:-1}
extra_args=${RUN_EXTRA_ARGS:-}

if [ -n "${PYTHON_BIN:-}" ]; then
  py_bin="$PYTHON_BIN"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  py_bin="${CONDA_PREFIX}/bin/python"
elif [ -x "${HOME}/miniconda3/envs/mappo-mpc/bin/python" ]; then
  py_bin="${HOME}/miniconda3/envs/mappo-mpc/bin/python"
elif command -v python >/dev/null 2>&1; then
  py_bin=python
else
  py_bin=python3
fi

cfg_file="configs/${task}.args"
if [ ! -f "$cfg_file" ]; then
  echo "[error] config file not found: $cfg_file" >&2
  exit 1
fi

cfg_args=$(sed -E 's/[[:space:]]*#.*$//' "$cfg_file" | grep -Ev '^[[:space:]]*$' | xargs)
cfg_args="$cfg_args $extra_args"
read -r -a cfg_tokens <<< "$cfg_args"

camera_control_mode=learned
sensor_grad_mode=full
policy_gray_mode=gray
train_camera_only=0
train_flight_only=0
for ((i=0; i<${#cfg_tokens[@]}; i++)); do
  tok="${cfg_tokens[$i]}"
  if [ "$tok" = "--camera_control_mode" ] && [ $((i+1)) -lt ${#cfg_tokens[@]} ]; then
    camera_control_mode="${cfg_tokens[$((i+1))]}"
  elif [ "$tok" = "--sensor_grad_mode" ] && [ $((i+1)) -lt ${#cfg_tokens[@]} ]; then
    sensor_grad_mode="${cfg_tokens[$((i+1))]}"
  elif [ "$tok" = "--policy_gray_mode" ] && [ $((i+1)) -lt ${#cfg_tokens[@]} ]; then
    policy_gray_mode="${cfg_tokens[$((i+1))]}"
  elif [ "$tok" = "--train_camera_only" ]; then
    train_camera_only=1
  elif [ "$tok" = "--train_flight_only" ]; then
    train_flight_only=1
  fi
done

phase=joint
[ "$train_camera_only" = "1" ] && phase=cameraonly
[ "$train_flight_only" = "1" ] && phase=flightonly
run_tag="gray_cam-${camera_control_mode}_grad-${sensor_grad_mode}_vision-${policy_gray_mode}_${phase}"

mkdir -p logs
date_tag=$(date +%Y-%m-%d-%H-%M-%S)
log_file="logs/${date_tag}-${task}-${run_tag}.log"

echo "config  : $cfg_file"
echo "run tag : $run_tag"
echo "python  : $py_bin"
echo "log     : $log_file"
echo "extra   : ${extra_args:-<none>}"

export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
ulimit -c unlimited 2>/dev/null || true

if [ "$log_to_file" = "1" ]; then
  "$py_bin" -u main_cuda.py $cfg_args > "$log_file" 2>&1
elif [ "$log_to_file" = "0" ]; then
  "$py_bin" -u main_cuda.py $cfg_args
else
  echo "[error] LOG_TO_FILE must be 0 or 1" >&2
  exit 1
fi
