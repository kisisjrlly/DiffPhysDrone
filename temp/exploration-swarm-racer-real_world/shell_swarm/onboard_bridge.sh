#gnome-terminal
#!/bin/bash 

tmux new-session -d -s bridge;
tmux set mouse on

tmux send-keys "sleep 1s" C-m 
tmux send-keys "source ~/ws_swarm_racer/racer/devel/setup.bash" C-m
tmux send-keys "roslaunch swarm_ros_bridge onboard_bridge.launch" C-m

tmux -2 attach-session -t bridge
