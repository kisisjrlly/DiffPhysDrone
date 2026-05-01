#!/bin/bash

# ===================== 你只需要改这一行 =====================
# path="checkpoint/2026-04-25-16-19-31/checkpoint0049.pth"
path="checkpoint/2026-04-30-22-31-31/checkpoint0024.pth"
# path="checkpoint/2026-04-26-11-24-40/checkpoint0034.pth"
# ============================================================

# 远程服务器配置
remote_user="root"
remote_ip="10.130.145.237"
remote_port="46936"
remote_base="/root/code/DiffPhysDrone"

# 本地基础路径
local_base="/home/zhaoguodong/work/code/DiffPhysDrone"

# 自动提取文件所在的文件夹
folder=$(dirname "$path")

# 本地要创建的目录
local_target_dir="$local_base/$folder"

# 自动创建目录（不存在就创建）
echo "创建本地目录：$local_target_dir"
mkdir -p "$local_target_dir"

# 拼接远程完整路径
remote_full_path="$remote_base/$path"

# 执行拷贝
echo "正在从远程下载：$remote_full_path"
scp -P $remote_port $remote_user@$remote_ip:$remote_full_path $local_target_dir/

echo "✅ 拷贝完成！"