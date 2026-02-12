#!/bin/bash

task=paper_gdac


date=$(date +%Y-%m-%d-%H-%M-%S)

python -u main_cuda.py $(cat configs/$task.args) > logs/$task-$date.log 2>&1