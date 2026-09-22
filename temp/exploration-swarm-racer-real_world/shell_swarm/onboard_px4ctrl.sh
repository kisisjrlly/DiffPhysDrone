#gnome-terminal
#!/bin/bash 

tmux new-session -d -s ctrl;
tmux set mouse on

tmux send-keys "sleep 1s" C-m 
tmux send-keys "source ~/ws_swarm_racer/lio/devel/setup.bash" C-m
tmux send-keys "roslaunch px4ctrl run_ctrl.launch" C-m

tmux -2 attach-session -t ctrl
