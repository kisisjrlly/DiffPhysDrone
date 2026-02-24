#!/bin/bash

# 运行脚本 (Run script)

# 设置要运行的任务名称 (Set the task to run)
task=paper_gdac

# 获取当前日期和时间，用于日志文件名 (Get current date and time for log file name)
date=$(date +%Y-%m-%d-%H-%M-%S)

# 运行主程序，读取对应的配置文件，并将输出重定向到日志文件
# (Run the main program, read the corresponding config file, and redirect output to a log file)
python -u main_cuda.py $(cat configs/$task.args) > logs/$task-$date.log 2>&1