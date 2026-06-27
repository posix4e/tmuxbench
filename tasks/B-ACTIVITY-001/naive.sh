# Naive agent: doesn't read the activity flag, guesses the focused window.
tmux list-windows -t a
printf a > "$REPORT"
