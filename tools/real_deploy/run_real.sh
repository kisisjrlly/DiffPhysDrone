#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

real_args_file=${REAL_ARGS_FILE:-"${script_dir}/real.args"}
log_to_file=${LOG_TO_FILE:-1}

if [ -n "${PYTHON_BIN:-}" ]; then
    py_bin="$PYTHON_BIN"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    py_bin="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    py_bin="python3"
elif command -v python >/dev/null 2>&1; then
    py_bin="python"
else
    echo "[real][error] cannot find python interpreter"
    echo "             set PYTHON_BIN=/path/to/python if needed"
    exit 1
fi

if [ ! -f "$real_args_file" ]; then
    echo "[real][error] args file not found: $real_args_file"
    exit 1
fi

if ! { [ -x "$py_bin" ] || command -v "$py_bin" >/dev/null 2>&1; }; then
    echo "[real][error] python executable not found: $py_bin"
    exit 1
fi

mapfile -d '' -t cfg_tokens < <(
    "$py_bin" - "$real_args_file" <<'PY'
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
tokens = []
for raw in path.read_text(encoding='utf-8').splitlines():
    line = raw.split('#', 1)[0].strip()
    if not line:
        continue
    tokens.extend(shlex.split(line))

if tokens:
    sys.stdout.write("\0".join(tokens))
    sys.stdout.write("\0")
PY
)

all_tokens=("${cfg_tokens[@]}" "$@")

checkpoint=""
args_file=""
arm_flag="false"
takeoff_flag="false"
for ((i=0; i<${#all_tokens[@]}; i++)); do
    token="${all_tokens[$i]}"
    if [ "$token" = "--checkpoint" ] && [ $((i + 1)) -lt ${#all_tokens[@]} ]; then
        checkpoint="${all_tokens[$((i + 1))]}"
    fi
    if [ "$token" = "--args-file" ] && [ $((i + 1)) -lt ${#all_tokens[@]} ]; then
        args_file="${all_tokens[$((i + 1))]}"
    fi
    if [ "$token" = "--arm" ]; then
        arm_flag="true"
    fi
    if [ "$token" = "--no-arm" ]; then
        arm_flag="false"
    fi
    if [ "$token" = "--auto-takeoff" ]; then
        takeoff_flag="true"
    fi
    if [ "$token" = "--no-auto-takeoff" ]; then
        takeoff_flag="false"
    fi
done

if [ -z "$checkpoint" ]; then
    echo "[real][error] --checkpoint is missing."
    echo "             Please edit: $real_args_file"
    exit 1
fi

if [[ "$checkpoint" == *REPLACE_ME* ]]; then
    echo "[real][error] --checkpoint still contains placeholder text:"
    echo "             $checkpoint"
    echo "             Please edit: $real_args_file"
    exit 1
fi

if [ -z "$args_file" ]; then
    args_file="configs/paper_final_full.args"
fi

checkpoint_tag=$(basename "${checkpoint}")
checkpoint_tag="${checkpoint_tag%.pth}"
checkpoint_tag="${checkpoint_tag// /_}"
run_tag="arm-${arm_flag}_takeoff-${takeoff_flag}_${checkpoint_tag}"

log_dir="${repo_root}/logs/real_deploy"
mkdir -p "$log_dir"
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
stdout_log="${log_dir}/${timestamp}-${run_tag}.log"

cmd=("$py_bin" -u tools/real_deploy/run_real_policy.py "${all_tokens[@]}")

echo "[real] repo_root       : $repo_root"
echo "[real] args_file       : $real_args_file"
echo "[real] project args    : $args_file"
echo "[real] checkpoint      : $checkpoint"
echo "[real] stdout log      : $stdout_log"
echo "[real] arm             : $arm_flag"
echo "[real] auto_takeoff    : $takeoff_flag"
printf '[real] command        : '
printf '%q ' "${cmd[@]}"
printf '\n'

if [ "$arm_flag" = "true" ]; then
    echo "[real][warn] --arm is enabled. Make sure props / safety workflow are ready."
fi

cd "$repo_root"

if [ "$log_to_file" = "1" ]; then
    "${cmd[@]}" 2>&1 | tee "$stdout_log"
elif [ "$log_to_file" = "0" ]; then
    "${cmd[@]}"
else
    echo "[real][error] invalid LOG_TO_FILE=$log_to_file (expected 0 or 1)"
    exit 1
fi
