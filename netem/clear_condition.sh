#!/usr/bin/env bash
# Remove any netem rule from the interface. Safe to run when nothing is applied.
#   bash netem/clear_condition.sh [interface]
set -uo pipefail
iface="${1:-${NETEM_IFACE:-}}"
if [[ -z "$iface" || "$iface" == "auto" ]]; then
  iface="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
fi
[[ -z "$iface" ]] && { echo "could not detect interface; pass it as the first argument" >&2; exit 1; }
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
$SUDO tc qdisc del dev "$iface" root 2>/dev/null || true
echo "[netem] $iface: cleared"
