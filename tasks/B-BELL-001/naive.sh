# Naive: guess a window without ever reading the bell flag -> misses the event.
tmux list-windows -t srv -F '#{window_name}' | head -1 > "$REPORT"
