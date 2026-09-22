#!/bin/bash

# 定义变量
USER="swarm2"
HOST="192.168.124.202"
PASSWORD="123"  # 请替换为你的密码

# 使用sshpass执行SSH命令
sshpass -p "$PASSWORD" ssh "$USER@$HOST"