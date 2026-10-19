#!/usr/bin/env bash
# Apply a named network condition to YOUR outgoing traffic using Linux tc/netem.
#
#   bash netem/apply_condition.sh <profile> [interface]
#   bash netem/apply_condition.sh lossy            # interface auto-detected
#   bash netem/apply_condition.sh clean            # same as clear_condition.sh
#
# Profiles (what is added to every packet LEAVING this machine):
#   clean        nothing (removes any rule)
#   moderate     delay 50ms +-10ms, 0.5% loss
#   high_jitter  delay 80ms +-40ms (normal distribution), no loss
#   lossy        delay 30ms +-5ms, 3% loss
#
# netem shapes EGRESS only, so a 50ms delay adds ~50ms to the round trip (not 100).
# State this in the Methodology. Needs root for `tc`; the script calls sudo itself.
# ALWAYS undo with clear_condition.sh - or use run_with_condition.sh which does it for you.
set -euo pipefail

profile="${1:-}"
iface="${2:-${NETEM_IFACE:-}}"

if [[ -z "$profile" ]]; then
  sed -n '2,15p' "$0"; exit 1
fi

detect_iface() {
  ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'
}
[[ -z "$iface" || "$iface" == "auto" ]] && iface="$(detect_iface)"
if [[ -z "$iface" ]]; then
  echo "could not detect the network interface; pass it explicitly (see: ip -br addr)" >&2; exit 1
fi

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
command -v tc >/dev/null || { echo "tc not found: sudo apt install iproute2" >&2; exit 1; }

case "$profile" in
  clean)
    $SUDO tc qdisc del dev "$iface" root 2>/dev/null || true
    echo "[netem] $iface: clean (no rule)"; exit 0 ;;
  moderate)    args="delay 50ms 10ms loss 0.5%" ;;
  high_jitter) args="delay 80ms 40ms distribution normal" ;;
  lossy)       args="delay 30ms 5ms loss 3%" ;;
  *) echo "unknown profile '$profile' (clean | moderate | high_jitter | lossy)" >&2; exit 1 ;;
esac

# 'replace' works whether or not a rule already exists (no "File exists" error)
# shellcheck disable=SC2086
$SUDO tc qdisc replace dev "$iface" root netem $args
echo "[netem] $iface: $profile -> netem $args"
$SUDO tc qdisc show dev "$iface" | head -n 2
