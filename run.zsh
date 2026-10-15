#!/usr/bin/env zsh
# Zsh wrapper for run.sh.

emulate -L bash
exec "$(dirname "$0:A")/run.sh" "$@"
