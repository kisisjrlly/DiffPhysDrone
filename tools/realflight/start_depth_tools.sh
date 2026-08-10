#!/bin/bash
# Start the D455 depth stream + rqt tools on the onboard computer.
# Safe to run over SSH: long-running processes are detached with setsid+nohup,
# so they survive the SSH session closing.

export ROS_DISTRO=noetic
source /opt/ros/noetic/setup.bash

pkill -f '[/]opt/ros/noetic/lib/nodelet/nodelet' 2>/dev/null || true
pkill -x realsense-viewer 2>/dev/null || true
pkill -f '[/]opt/ros/noetic/bin/roslaunch realsense2_camera' 2>/dev/null || true
sleep 1

if ! pgrep -x rosmaster >/dev/null; then
  setsid nohup roscore >/tmp/roscore.log 2>&1 < /dev/null &
  sleep 4
fi

setsid nohup roslaunch realsense2_camera rs_camera.launch \
  enable_depth:=true enable_infra1:=false enable_infra2:=false enable_color:=false \
  depth_width:=640 depth_height:=480 depth_fps:=15 \
  enable_emitter:=false initial_reset:=false \
  >/tmp/rs_camera.log 2>&1 < /dev/null &
echo "camera bg pid: $!"

for i in $(seq 1 25); do
  if timeout 2 rostopic list 2>/dev/null | grep -q '/camera/depth/image_rect_raw$'; then
    echo "depth topic up after ${i}s"
    break
  fi
  sleep 1
done

echo '--- depth topic ---'
rostopic list 2>/dev/null | grep -E '/camera/depth/image_rect_raw$' || echo NO_TOPIC
echo '--- hz (5s) ---'
timeout 5 rostopic hz /camera/depth/image_rect_raw 2>&1 | tail -n 2

if [ -n "${DISPLAY:-}" ]; then
  setsid nohup rqt_image_view >/tmp/rqt_image_view.log 2>&1 < /dev/null &
  setsid nohup rqt_reconfigure >/tmp/rqt_reconfigure.log 2>&1 < /dev/null &
  echo "rqt GUIs launched on DISPLAY=$DISPLAY"
else
  echo "no DISPLAY; rqt not launched"
fi

# Browser-based live depth + parameter control (no GUI client needed on host)
if [ -x /home/xgg/start_depth_web.sh ]; then
  bash /home/xgg/start_depth_web.sh
fi
