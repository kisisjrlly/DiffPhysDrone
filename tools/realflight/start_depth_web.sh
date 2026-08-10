#!/bin/bash
# Start the depth web viewer + parameter control tool on the onboard computer.
# After it starts, open http://<onboard-ip>:8090 from any browser on the LAN.

export ROS_DISTRO=noetic
source /opt/ros/noetic/setup.bash

pkill -9 -f 'depth_web_tool[.]py --port' 2>/dev/null || true
if pgrep -f 'depth_web_tool.py' >/dev/null; then
  echo "depth_web_tool already running"
else
  setsid nohup python3 /home/xgg/depth_web_tool.py --port 8090 \
    >/tmp/depth_web_tool.log 2>&1 < /dev/null &
  echo "depth_web_tool started (pid $!)"
fi

for i in $(seq 1 10); do
  if ss -tln 2>/dev/null | grep -q ':8090'; then
    echo "listening on http://0.0.0.0:8090"
    break
  fi
  sleep 1
done
tail -5 /tmp/depth_web_tool.log 2>/dev/null
