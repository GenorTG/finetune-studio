#!/usr/bin/env zsh
# Zsh wrapper for the Finetune Studio installer.

emulate -L bash
exec "$(dirname "$0:A")/install.sh" "$@"