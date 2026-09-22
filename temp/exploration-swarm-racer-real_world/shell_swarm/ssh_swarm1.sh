#!/bin/bash

# 定义变量
USER="swarm1"
HOST="192.168.124.201"
PASSWORD="123"  # 请替换为你的密码

# 使用sshpass执行SSH命令
sshpass -p "$PASSWORD" ssh "$USER@$HOST"