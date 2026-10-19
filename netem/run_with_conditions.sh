#!/usr/bin/env bash
# Apply a condition, run a command, and ALWAYS remove the condition afterwards -
# even if the command crashes or you press Ctrl-C (that is what `trap` is for).
# Run it as your normal user (venv active): it uses sudo only for the tc commands.
#
#   bash netem/run_with_condition.sh <profile> -- <command...>
#   bash netem/run_with_condition.sh lossy -- python client/run_batch.py --person A --profile lossy --repeats 3
#
# Interface: auto-detected, or set NETEM_IFACE=eth0 before the command.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

profile="${1:-}"
[[ -z "$profile" || "${2:-}" != "--" ]] && { echo "usage: $0 <profile> -- <command...>" >&2; exit 1; }
shift 2
[[ $# -eq 0 ]] && { echo "no command given after --" >&2; exit 1; }

cleanup() { bash "$here/clear_condition.sh" "${NETEM_IFACE:-auto}" >/dev/null 2>&1 || true
            echo "[netem] condition removed"; }
trap cleanup EXIT INT TERM

bash "$here/apply_condition.sh" "$profile" "${NETEM_IFACE:-auto}" || exit 1
"$@"
exit $?
