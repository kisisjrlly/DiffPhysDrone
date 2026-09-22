#gnome-terminal
#!/bin/bash 

tmux new-session -d -s lio;
tmux set mouse on

tmux split-window -h 
tmux split-window -v 
tmux select-pane -t 0
tmux split-window -v 

tmux select-pane -t 0
# CMD: launch mid360
tmux send-keys "sleep 1s" C-m 
tmux send-keys "source ~/tools/ws_livox_ros_driver2/devel/setup.bash" C-m
tmux send-keys "sleep 1s" C-m 
tmux send-keys "roslaunch livox_ros_driver2 msg_MID360.launch" C-m 

tmux select-pane -t 1
# CMD: launch mavros px4
tmux send-keys "sleep 5s" C-m 
tmux send-keys "roslaunch mavros px4.launch" C-m 

tmux select-pane -t 2
# CMD: launch fast-lio2
tmux send-keys "sleep 10s" C-m 
tmux send-keys "source ~/ws_swarm_racer/lio/devel/setup.bash" C-m
tmux send-keys "sleep 1s" C-m
tmux send-keys "roslaunch fast_lio mapping_mid360.launch" C-m

tmux select-pane -t 3
# 4 cmd
tmux send-keys "sleep 15s" C-m 
tmux send-keys "source ~/ws_swarm_racer/lio/devel/setup.bash" C-m
tmux send-keys "sleep 1s" C-m
tmux send-keys "roslaunch ekf fast_lio_and_imu.launch" C-m

tmux -2 attach-session -t lio
