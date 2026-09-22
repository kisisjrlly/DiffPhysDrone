#gnome-terminal
#!/bin/bash

tmux new-session -d -s split_window;
tmux set mouse on

tmux split-window -v
tmux select-pane -t 0

tmux select-pane -t 0
tmux split-window -v

tmux select-pane -t 1
tmux split-window -h

tmux select-pane -t 0
tmux resize-pane -U 10

tmux select-pane -t 3
tmux resize-pane -U 10

tmux -2 attach-session -t split_window