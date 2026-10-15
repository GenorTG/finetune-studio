#!/usr/bin/env bash
# Thin wrapper: the real installer lives at the repo root and defaults to a
# systemd --user unit (no sudo). Pass --system for the old root-level unit.
#
#   bash scripts/install-service.sh            # user unit, no sudo
#   bash scripts/install-service.sh --restart  # restart it
#   bash scripts/install-service.sh --status

set -euo pipefail
exec bash "$(dirname "$(readlink -f "$0")")/../install-service.sh" "$@"
