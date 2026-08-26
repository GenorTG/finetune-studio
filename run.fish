#!/usr/bin/env fish
# Fish wrapper for run.sh.

bash (dirname (status -f))/run.sh $argv
