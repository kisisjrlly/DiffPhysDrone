#gnome-terminal
#!/bin/bash

tmux new-session -d -s gcs;
tmux set mouse on

tmux split-window -v
tmux select-pane -t 0

tmux select-pane -t 0
# CMD: gcs node
tmux send-keys "sleep 1s" C-m
tmux send-keys "source ~/ws_swarm_exp/gcs_real/devel/setup.bash" C-m
tmux send-keys "sleep 1s" C-m
tmux send-keys "roslaunch exploration_manager ground_station.launch" C-m

tmux select-pane -t 1
# CMD: bridge
tmux send-keys "sleep 1s" C-m
tmux send-keys "source ~/ws_swarm_exp/gcs_real/devel/setup.bash" C-m
tmux send-keys "sleep 3s" C-m
tmux send-keys "roslaunch swarm_ros_bridge gcs_bridge.launch" C-m

tmux -2 attach-session -t gcs