#!/bin/bash

# 运行脚本 (Run script)

set -euo pipefail
# set -x

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

# 设置要运行的任务名称 (Set the task to run)
# 可通过环境变量覆盖，例如：TASK=paper_intent_lqr bash run.sh
task=${TASK:-paper_final_full}

# 可选相机档位叠加配置：
#   CAM_PROFILE=low|high|ultra
# 示例：
#   TASK=paper_final_full CAM_PROFILE=ultra bash run.sh
# 默认使用 high（吞吐与保真折中）；若要严格遵循任务配置，请显式传空 CAM_PROFILE。
cam_profile=${CAM_PROFILE:-low}
# cam_profile=${CAM_PROFILE:-low}

# 日志输出模式:
# - LOG_TO_FILE=1 (默认): 输出到 logs/<task>-<time>.log
# - LOG_TO_FILE=0       : 仅输出到终端，不写入日志文件
# 示例:
#   LOG_TO_FILE=1 bash run.sh
#   LOG_TO_FILE=0 bash run.sh
log_to_file=${LOG_TO_FILE:-1}

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

mkdir -p logs

# 获取当前日期和时间，用于日志文件名 (Get current date and time for log file name)
date=$(date +%Y-%m-%d-%H-%M-%S)
log_file="logs/$task-$date.log"
echo "log file: $log_file"

# 运行主程序，读取对应的配置文件，并将输出重定向到日志文件
# (Run the main program, read the corresponding config file, and redirect output to a log file)
# 支持：
# 1) 空行
# 2) 整行注释（以 # 开头）
# 3) 行尾注释（参数后跟 # 注释）
cfg_args=""
for f in "${cfg_files[@]}"; do
	part=$(sed -E 's/[[:space:]]*#.*$//' "$f" | grep -Ev '^[[:space:]]*$' | xargs)
	cfg_args="$cfg_args $part"
done

echo "using config files: ${cfg_files[*]}"

# CUDA 显存分配器：降低碎片导致的 OOM 概率（不改变训练配置/模型）
# 注：某些 PyTorch/CUDA 组合下 expandable_segments 可能触发内部断言，默认不开启。
# export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128,garbage_collection_threshold:0.8"}


if [ "$log_to_file" = "1" ]; then
	python -u main_cuda.py $cfg_args > "$log_file" 2>&1
elif [ "$log_to_file" = "0" ]; then
	python -u main_cuda.py $cfg_args > res.log 2>&1
	# python -u main_cuda.py $cfg_args
else
	echo "[error] invalid LOG_TO_FILE=$log_to_file (expected 0 or 1)"
	exit 1
fi