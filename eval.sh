#!/bin/bash

# 评估脚本：仅推理 + rerun 可视化（不训练、不算 loss、不上 wandb）

set -euo pipefail

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

# 任务名（对应 configs/<task>.args）
task=${TASK:-paper_final_full}

# 可选叠加相机档位（与 run.sh 一致）
cam_profile=${CAM_PROFILE:-}

# 评估 checkpoint（可通过环境变量覆盖）
# ckpt_path=${CKPT:-checkpoint/2026-03-18-22-23-06/checkpoint0049.pth}
# ckpt_path=${CKPT:-checkpoint/2026-03-20-10-05-24/checkpoint0000.pth}
# ckpt_path=${CKPT:-checkpoint/2026-03-19-17-20-34/checkpoint0049.pth} 往后退
# ckpt_path=${CKPT:-checkpoint/2026-03-20-21-45-04/checkpoint0049.pth} # 向后退
# ckpt_path=${CKPT:-checkpoint/2026-03-24-22-56-19/checkpoint0045.pth} 
# ckpt_path=${CKPT:-checkpoint/2026-03-31-21-03-07/checkpoint0005.pth} 
# ckpt_path=${CKPT:-checkpoint/2026-04-15-11-15-54/checkpoint0011.pth} # 继续往后退
ckpt_path=${CKPT:-checkpoint/2026-04-02-10-51-57/checkpoint0049.pth}


# 评估 episode 数（默认 1）
eval_episodes=${EVAL_EPISODES:-10}

# 评估 batch 大小（默认 1，只看单机/单轨迹结果）
eval_batch_size=${EVAL_BATCH_SIZE:-1}

# 日志输出模式:
# - LOG_TO_FILE=1 (默认): 输出到 logs/eval-<task>-<time>.log
# - LOG_TO_FILE=0       : 仅输出到终端
log_to_file=${LOG_TO_FILE:-0}

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

cfg_files=("$cfg_file")
if [ -n "$cam_profile" ]; then
	profile_file="configs/cam_${cam_profile}.args"
	if [ ! -f "$profile_file" ]; then
		echo "[error] camera profile file not found: $profile_file"
		echo "        expected CAM_PROFILE in {low,high,ultra}"
		exit 1
	fi
	cfg_files+=("$profile_file")
fi

if [ ! -f "$ckpt_path" ]; then
	echo "[error] checkpoint not found: $ckpt_path"
	exit 1
fi

mkdir -p logs

date=$(date +%Y-%m-%d-%H-%M-%S)
log_file="logs/eval-${task}-${date}.log"
echo "log file: $log_file"

echo "using config files: ${cfg_files[*]}"
echo "using checkpoint  : $ckpt_path"
echo "eval episodes     : $eval_episodes"
echo "eval batch_size   : $eval_batch_size"

if ! command -v "$py_bin" >/dev/null 2>&1; then
	echo "[error] python executable not found: $py_bin"
	echo "        set PYTHON_BIN=/path/to/python if needed"
	exit 1
fi

# 支持：空行 / 整行注释 / 行尾注释
cfg_args=""
for f in "${cfg_files[@]}"; do
	part=$(sed -E 's/[[:space:]]*#.*$//' "$f" | grep -Ev '^[[:space:]]*$' | xargs)
	cfg_args="$cfg_args $part"
done

# 强制 eval 语义：
# - 使用指定 checkpoint
# - 禁用 wandb
# - 仅运行 eval.py 前向推理
common_cmd="python -u eval.py $cfg_args --resume $ckpt_path --vis_enable --wandb_disabled --eval_episodes $eval_episodes --batch_size $eval_batch_size --timesteps 300"
common_cmd="$py_bin -u eval.py $cfg_args --resume $ckpt_path --vis_enable --wandb_disabled --eval_episodes $eval_episodes --batch_size $eval_batch_size --timesteps 300"

if [ "$log_to_file" = "1" ]; then
	eval "$common_cmd" > "$log_file" 2>&1
elif [ "$log_to_file" = "0" ]; then
	eval "$common_cmd"
else
	echo "[error] invalid LOG_TO_FILE=$log_to_file (expected 0 or 1)"
	exit 1
fi
