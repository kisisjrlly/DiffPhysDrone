#!/bin/bash
set -euo pipefail

# Minimal eval wrapper. Usage:
#   CKPT=checkpoint/.../checkpointXXXX.pth bash eval.sh
# Optional:
#   MODE=randfix CKPT=checkpoint/.../checkpointXXXX.pth bash eval.sh
#   MODE=flightonly CKPT=checkpoint/.../checkpointXXXX.pth bash eval.sh
#   TRACE=1 CKPT=checkpoint/.../checkpointXXXX.pth bash eval.sh
#   EVAL_EXTRA_ARGS="--scenarios glare --sun_glare_eval_slot right" bash eval.sh

export all_proxy=${all_proxy:-http://127.0.0.1:7890}

base_task=${BASE_TASK:-slit_active_sensing}
mode=${MODE:-fix}
config_override=${CONFIG:-}
task_override=${TASK:-}
ckpt_path=${CKPT:-}
eval_episodes=${EVAL_EPISODES:-10}
vis_episode_idx=${VIS_EPISODE_IDX:--1}
eval_batch_size=${EVAL_BATCH_SIZE:-1}
vis_enable=${VIS_ENABLE:-1}
eval_extra_args=${EVAL_EXTRA_ARGS:-}
eval_trace_csv=${EVAL_TRACE_CSV:-}
eval_episode_csv=${EVAL_EPISODE_CSV:-}
log_to_file=${LOG_TO_FILE:-0}
dry_run=${DRY_RUN:-0}

case "$mode" in
	ours|fixed|randfix|nondiff|flightonly|learned_detached)
		;;
	fix)
		mode="fixed"
		;;
	fixed_random|fixed_random_static)
		mode="randfix"
		;;
	flight_only|flight)
		mode="flightonly"
		;;
	*)
		echo "[error] unsupported MODE=$mode (expected ours|fixed|randfix|nondiff|flightonly|learned_detached)" >&2
		exit 1
		;;
esac

default_task_for_mode() {
	case "$1" in
		learned_detached|flightonly)
			echo "slit_active_sensing_auto_flightonly"
			;;
		*)
			echo "${base_task}_auto_$1"
			;;
	esac
}

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

if [ -n "$config_override" ]; then
	cfg_file="$config_override"
	task=$(basename "${cfg_file%.args}")
elif [ -n "$task_override" ]; then
	task="$task_override"
	cfg_file="configs/${task}.args"
else
	task="$(default_task_for_mode "$mode")"
	cfg_file="configs/${task}.args"
fi

if [ ! -f "$cfg_file" ]; then
	echo "[error] config file not found: $cfg_file"
	echo "        generate it with: RUN_TRAIN=0 MODES=\"ours fixed randfix nondiff\" bash run_train_modes.sh"
	exit 1
fi

cfg_args=$(sed -E 's/[[:space:]]*#.*$//' "$cfg_file" | grep -Ev '^[[:space:]]*$' | xargs)
if [ -n "$cfg_args" ]; then
	cfg_args=$(python3 - "$cfg_args" <<'PY'
import shlex
import sys

tokens = shlex.split(sys.argv[1])
out = []
skip = False
for tok in tokens:
    if skip:
        skip = False
        continue
    if tok == "--resume":
        skip = True
        continue
    if tok.startswith("--resume="):
        continue
    out.append(tok)
print(" ".join(shlex.quote(tok) for tok in out))
PY
)
fi
vis_args=""
if [ "$vis_enable" = "1" ]; then
	vis_args="--vis_enable"
elif [ "$vis_enable" != "0" ]; then
	echo "[error] invalid VIS_ENABLE=$vis_enable"
	exit 1
fi
csv_args=""
trace=${TRACE:-0}
if [ "$trace" = "1" ]; then
	[ -z "$eval_trace_csv" ] && eval_trace_csv="logs/eval-${task}-${mode}-trace.csv"
	[ -z "$eval_episode_csv" ] && eval_episode_csv="logs/eval-${task}-${mode}-episodes.csv"
elif [ "$trace" != "0" ]; then
	echo "[error] invalid TRACE=$trace (expected 0 or 1)" >&2
	exit 1
fi
[ -n "$eval_trace_csv" ] && csv_args="$csv_args --eval_trace_csv $eval_trace_csv"
[ -n "$eval_episode_csv" ] && csv_args="$csv_args --eval_episode_csv $eval_episode_csv"

mkdir -p logs
log_file="logs/eval-${task}-$(date +%Y-%m-%d-%H-%M-%S).log"
cmd="$py_bin -u eval.py $cfg_args $eval_extra_args --resume $ckpt_path $vis_args --wandb_disabled --eval_episodes $eval_episodes --vis_episode_idx $vis_episode_idx --batch_size $eval_batch_size $csv_args"

echo "using config     : $cfg_file"
echo "mode             : $mode"
echo "using checkpoint : $ckpt_path"
echo "eval episodes    : $eval_episodes"
echo "eval extra args  : ${eval_extra_args:-<none>}"

if [ "$dry_run" = "1" ]; then
	echo "dry run command  : $cmd"
elif [ "$dry_run" != "0" ]; then
	echo "[error] invalid DRY_RUN=$dry_run"
	exit 1
elif [ "$log_to_file" = "1" ]; then
	eval "$cmd" > "$log_file" 2>&1
	echo "log file: $log_file"
elif [ "$log_to_file" = "0" ]; then
	eval "$cmd"
else
	echo "[error] invalid LOG_TO_FILE=$log_to_file"
	exit 1
fi
