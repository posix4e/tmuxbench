# A "naive" agent that does unrelated work and MISSES the death event:
# it never inspects the worker pane and never respawns it.
# Used to validate that the missed-event metric catches an unhandled event.
tmux new-window -t svc -n other
tmux send-keys -t svc:other 'echo working' Enter
