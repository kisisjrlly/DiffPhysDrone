#gnome-terminal
#!/bin/bash 

tmux new-session -d -s exploration;
tmux set mouse on

tmux send-keys "sleep 1s" C-m 
tmux send-keys "source ~/ws_swarm_racer/racer/devel/setup.bash" C-m
tmux send-keys "roslaunch exploration_manager swarm_exploration_realworld.launch" C-m

tmux -2 attach-session -t exploration
