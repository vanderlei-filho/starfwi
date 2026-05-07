#!/usr/bin/env bash
# imds_watchdog.sh — polls the EC2 instance metadata service for a spot
# interruption notice and sends SIGUSR1 to the local StarFWI MPI processes
# when one is detected.
#
# Usage: imds_watchdog.sh <poll_interval_seconds>
#   poll_interval_seconds  How often to query IMDS (default: 5)
#
# The script sends SIGUSR1 to every process named "starfwi-fwi" on this node.
# StarFWI's signal handler sets g_interrupt_requested, which is checked
# collectively at each shot boundary via MPI_Allreduce.
#
# Launched automatically by HPC@Cloud alongside mpirun.

set -euo pipefail

POLL_INTERVAL=${1:-5}
IMDS_URL="http://169.254.169.254/latest/meta-data/spot/termination-time"

echo "[imds_watchdog] Started (poll interval: ${POLL_INTERVAL}s)"

while true; do
    if curl -s -f --max-time 1 "${IMDS_URL}" > /dev/null 2>&1; then
        echo "[imds_watchdog] Spot interruption notice received — sending SIGUSR1 to starfwi-fwi processes"
        pkill -USR1 -x starfwi-fwi || true
        echo "[imds_watchdog] Signal sent. Exiting."
        exit 0
    fi
    sleep "${POLL_INTERVAL}"
done
