#!/usr/bin/env bash
# run_experiment.sh — StarFWI experiment runner for IC2E 2026 evaluation
#
# Handles the full lifecycle of each experiment: launching the watchdog (for
# real-spot runs), running mpirun, parsing output, and appending results to
# per-experiment CSV files under $RESULTS_DIR.
#
# Usage:
#   run_experiment.sh --mode MODE [options]
#
# Modes:
#   baseline        Run without checkpointing; establishes the reference time.
#   overhead-sweep  Sweep checkpoint interval (shots); measure normalized time.
#   flush-scaling   Sweep MPI rank count; measure flush time and checkpoint size.
#                   Requires StarFWI to emit [METRIC] lines — see note below.
#   recovery        Inject spot failure via hpcac; measure end-to-end recovery.
#   elastic         Restart on a different rank count; measure time-to-solution.
#   cost-perf       Three elasticity policies; measure time + estimated AWS cost.
#
# NOTE on [METRIC] output:
#   flush-scaling expects StarFWI to print lines of the form:
#     [METRIC] flush_time_ms=1234 checkpoint_size_mb=56
#   after each call to starpu_mpi_checkpoint_flush_to_storage().
#   This requires a small addition to main_fwi.cpp (see TODO below).
#
# Environment / overrides (all have defaults):
#   BINARY, HOSTFILE, SEGY, OBSERVED_DIR, CHECKPOINT_DIR, RESULTS_DIR,
#   N_NODES, N_RANKS, N_SHOTS, N_TIMESTEPS, INSTANCE_TYPE, GPU_ENABLED,
#   HPCAC_CLUSTER_ID, MPIRUN

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

BINARY="${BINARY:-./build/bin/starfwi-fwi}"
MPIRUN="${MPIRUN:-mpirun}"
HOSTFILE="${HOSTFILE:-./hostfile}"
SEGY="${SEGY:-./data/marmousi2.segy}"
OBSERVED_DIR="${OBSERVED_DIR:-./data/observed}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/shared/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
LOGS_DIR="${RESULTS_DIR}/logs"

N_NODES="${N_NODES:-2}"
N_RANKS="${N_RANKS:-8}"
N_SHOTS="${N_SHOTS:-20}"
N_TIMESTEPS="${N_TIMESTEPS:-1000}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.2xlarge}"
GPU_ENABLED="${GPU_ENABLED:-0}"

# Used for recovery/cost-perf modes — the HPC@Cloud cluster to inject failures into
HPCAC_CLUSTER_ID="${HPCAC_CLUSTER_ID:-}"
# Node private IP to terminate for controlled failure injection
FAILURE_NODE_IP="${FAILURE_NODE_IP:-}"

MODE=""

# ─── Argument parsing ────────────────────────────────────────────────────────

usage() {
    grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)             MODE="$2";               shift 2 ;;
        --binary)           BINARY="$2";             shift 2 ;;
        --hostfile)         HOSTFILE="$2";           shift 2 ;;
        --segy)             SEGY="$2";               shift 2 ;;
        --observed-dir)     OBSERVED_DIR="$2";       shift 2 ;;
        --checkpoint-dir)   CHECKPOINT_DIR="$2";     shift 2 ;;
        --results-dir)      RESULTS_DIR="$2";        shift 2 ;;
        --n-nodes)          N_NODES="$2";            shift 2 ;;
        --n-ranks)          N_RANKS="$2";            shift 2 ;;
        --n-shots)          N_SHOTS="$2";            shift 2 ;;
        --n-timesteps)      N_TIMESTEPS="$2";        shift 2 ;;
        --instance-type)    INSTANCE_TYPE="$2";      shift 2 ;;
        --gpu)              GPU_ENABLED=1;           shift 1 ;;
        --cluster-id)       HPCAC_CLUSTER_ID="$2";  shift 2 ;;
        --failure-node-ip)  FAILURE_NODE_IP="$2";   shift 2 ;;
        --help|-h)          usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ -z "$MODE" ]] && { echo "Error: --mode is required." >&2; usage; }

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# ─── Helpers ─────────────────────────────────────────────────────────────────

log() { echo "[run_experiment][$(date -u +%H:%M:%S)] $*"; }

# Generate a unique run ID based on timestamp
run_id() { date -u +%Y%m%dT%H%M%SZ; }

# Write CSV header if the file does not yet exist
csv_header() {
    local file="$1"; shift
    [[ -f "$file" ]] || echo "$*" > "$file"
}

# Append a CSV row (args are already comma-separated fields)
csv_append() {
    local file="$1"; shift
    echo "$*" >> "$file"
}

# Parse "Total time: X.XXX s" from a log file; echoes the value
parse_total_time() {
    grep -oP 'Total time: \K[0-9]+\.[0-9]+' "$1" | tail -1
}

# Parse "[METRIC] flush_time_ms=X checkpoint_size_mb=Y" lines; echoes
# space-separated pairs of flush_time_s and checkpoint_size_mb
parse_flush_metrics() {
    grep '\[METRIC\]' "$1" | while IFS= read -r line; do
        local flush_ms cp_mb
        flush_ms=$(echo "$line" | grep -oP 'flush_time_ms=\K[0-9]+' || echo "")
        cp_mb=$(echo "$line"    | grep -oP 'checkpoint_size_mb=\K[0-9.]+' || echo "")
        echo "${flush_ms} ${cp_mb}"
    done
}

# Parse "[starfwi-fwi] Resuming from shot X of Y" — echoes X
parse_resume_shot() {
    grep -oP 'Resuming from shot \K[0-9]+' "$1" | tail -1
}

# Build the base mpirun command
mpirun_cmd() {
    local n_ranks="$1"; shift
    echo "$MPIRUN --hostfile $HOSTFILE -np $n_ranks $BINARY \
        --shots $N_SHOTS \
        --observed-dir $OBSERVED_DIR \
        $*"
}

# Launch the IMDS watchdog on every host in the hostfile in the background.
# Returns immediately; watchdog PIDs are tracked for cleanup.
launch_watchdogs() {
    log "Launching IMDS watchdog on all nodes..."
    while IFS= read -r host; do
        [[ -z "$host" || "$host" == \#* ]] && continue
        ssh "$host" "nohup $(pwd)/scripts/imds_watchdog.sh 5 &>/tmp/imds_watchdog.log &" &
    done < "$HOSTFILE"
    log "Watchdogs launched."
}

# Compute overhead % relative to a baseline time file
compute_overhead() {
    local total_time="$1"
    local baseline_file="${RESULTS_DIR}/baseline_time.txt"
    if [[ -f "$baseline_file" ]]; then
        local base
        base=$(cat "$baseline_file")
        echo "scale=2; ($total_time - $base) / $base * 100" | bc
    else
        echo "N/A"
    fi
}

# Query the current spot price for $INSTANCE_TYPE in the configured region.
# Echoes the price as a float, or "N/A" if unavailable.
query_spot_price() {
    aws ec2 describe-spot-price-history \
        --instance-types "$INSTANCE_TYPE" \
        --product-descriptions "Linux/UNIX" \
        --max-items 1 \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "N/A"
}

# Estimate cost in USD: instance_count * hours * price_per_hour
estimate_cost() {
    local n_instances="$1"
    local elapsed_s="$2"
    local price_per_hour="$3"
    if [[ "$price_per_hour" == "N/A" ]]; then
        echo "N/A"
    else
        echo "scale=4; $n_instances * ($elapsed_s / 3600) * $price_per_hour" | bc
    fi
}

# ─── Mode: baseline ──────────────────────────────────────────────────────────
# Run without checkpointing; record total execution time as the reference.

run_baseline() {
    local id; id=$(run_id)
    local logfile="${LOGS_DIR}/baseline_${id}.log"
    local csv="${RESULTS_DIR}/baseline.csv"

    csv_header "$csv" \
        "run_id,timestamp,n_nodes,n_ranks,n_shots,n_timesteps,instance_type,gpu_enabled,total_time_s"

    log "Mode: baseline | ranks=$N_RANKS shots=$N_SHOTS timesteps=$N_TIMESTEPS"

    $MPIRUN --hostfile "$HOSTFILE" -np "$N_RANKS" "$BINARY" \
        --shots "$N_SHOTS" \
        --observed-dir "$OBSERVED_DIR" \
        "$SEGY" "$N_TIMESTEPS" \
        2>&1 | tee "$logfile"

    local t
    t=$(parse_total_time "$logfile")
    [[ -z "$t" ]] && { log "ERROR: could not parse total time from $logfile"; exit 1; }

    # Save baseline time for overhead computations
    echo "$t" > "${RESULTS_DIR}/baseline_time.txt"

    csv_append "$csv" \
        "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$N_NODES,$N_RANKS,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$GPU_ENABLED,$t"

    log "Baseline complete: ${t}s → ${csv}"
}

# ─── Mode: overhead-sweep ────────────────────────────────────────────────────
# Sweep checkpoint interval from 1 shot to N_SHOTS (powers of 2).
# Measures normalized execution time relative to the baseline.

run_overhead_sweep() {
    local csv="${RESULTS_DIR}/overhead_sweep.csv"

    csv_header "$csv" \
        "run_id,timestamp,n_nodes,n_ranks,n_shots,n_timesteps,instance_type,gpu_enabled,checkpoint_interval_shots,total_time_s,overhead_pct"

    [[ ! -f "${RESULTS_DIR}/baseline_time.txt" ]] && \
        log "WARNING: no baseline_time.txt found — overhead_pct will be N/A"

    local intervals=(1 2 4 8 16 "$N_SHOTS")

    for k in "${intervals[@]}"; do
        [[ "$k" -gt "$N_SHOTS" ]] && continue
        local id; id=$(run_id)
        local logfile="${LOGS_DIR}/overhead_sweep_k${k}_${id}.log"

        log "overhead-sweep: checkpoint_interval=$k shots"

        $MPIRUN --hostfile "$HOSTFILE" -np "$N_RANKS" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            --checkpoint-dir "${CHECKPOINT_DIR}/sweep_k${k}_${id}" \
            --checkpoint-interval "$k" \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile"

        local t; t=$(parse_total_time "$logfile")
        local pct; pct=$(compute_overhead "$t")

        csv_append "$csv" \
            "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$N_NODES,$N_RANKS,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$GPU_ENABLED,$k,$t,$pct"

        log "  interval=$k total=${t}s overhead=${pct}%"
    done

    log "overhead-sweep complete → ${csv}"
}

# ─── Mode: flush-scaling ─────────────────────────────────────────────────────
# Sweep MPI rank count (1, 2, 4, 8, 16); measure flush time and checkpoint
# size per rank count.
#
# REQUIRES: StarFWI must print the following line after each flush:
#   [METRIC] flush_time_ms=<ms> checkpoint_size_mb=<mb>
# TODO: Add this to main_fwi.cpp after starpu_mpi_checkpoint_flush_to_storage().

run_flush_scaling() {
    local csv="${RESULTS_DIR}/flush_scaling.csv"

    csv_header "$csv" \
        "run_id,timestamp,n_ranks,n_shots,n_timesteps,instance_type,gpu_enabled,checkpoint_interval_shots,flush_time_s,checkpoint_size_mb"

    local rank_counts=(1 2 4 8 16)
    local cp_interval=4   # flush every 4 shots; one flush total if N_SHOTS=4

    for nr in "${rank_counts[@]}"; do
        [[ "$nr" -gt "$N_RANKS" ]] && break
        local id; id=$(run_id)
        local logfile="${LOGS_DIR}/flush_scaling_r${nr}_${id}.log"

        log "flush-scaling: n_ranks=$nr"

        $MPIRUN --hostfile "$HOSTFILE" -np "$nr" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            --checkpoint-dir "${CHECKPOINT_DIR}/flush_r${nr}_${id}" \
            --checkpoint-interval "$cp_interval" \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile"

        # Parse [METRIC] lines; average over all flushes in this run
        local sum_ms=0 sum_mb=0 count=0
        while IFS=' ' read -r flush_ms cp_mb; do
            [[ -z "$flush_ms" ]] && continue
            sum_ms=$(echo "$sum_ms + $flush_ms" | bc)
            sum_mb=$(echo "$sum_mb + $cp_mb"    | bc)
            ((count++)) || true
        done < <(parse_flush_metrics "$logfile")

        if [[ "$count" -eq 0 ]]; then
            log "  WARNING: no [METRIC] lines found in $logfile — binary needs update"
            csv_append "$csv" \
                "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$nr,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$GPU_ENABLED,$cp_interval,N/A,N/A"
        else
            local avg_s avg_mb
            avg_s=$(echo "scale=3; $sum_ms / $count / 1000" | bc)
            avg_mb=$(echo "scale=2; $sum_mb / $count"        | bc)
            csv_append "$csv" \
                "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$nr,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$GPU_ENABLED,$cp_interval,$avg_s,$avg_mb"
            log "  n_ranks=$nr avg_flush=${avg_s}s avg_size=${avg_mb}MB"
        fi
    done

    log "flush-scaling complete → ${csv}"
}

# ─── Mode: recovery ──────────────────────────────────────────────────────────
# Run a job, inject a failure via hpcac test-failure, measure end-to-end
# recovery time across three scenarios: same-size, scale-down, scale-up.
#
# Timing columns:
#   t_failure_inject  — wall time when test-failure was issued
#   t_job_resumed     — wall time when StarFWI prints "Resuming from shot"
#   total_recovery_s  — t_job_resumed - t_failure_inject
#   work_loss_shots   — shots that must be re-executed (from log)
#
# Note: finer breakdown (provisioning / EFS init / restore) requires
# HPC@Cloud to emit structured timing output — left as future instrumentation.

run_recovery() {
    [[ -z "$HPCAC_CLUSTER_ID" ]] && { log "ERROR: --cluster-id required for recovery mode"; exit 1; }
    [[ -z "$FAILURE_NODE_IP" ]]  && { log "ERROR: --failure-node-ip required for recovery mode"; exit 1; }

    local csv="${RESULTS_DIR}/recovery.csv"

    csv_header "$csv" \
        "run_id,timestamp,restart_scenario,n_ranks_before,n_ranks_after,n_shots,n_timesteps,instance_type,t_failure_inject,t_job_resumed,total_recovery_s,work_loss_shots,resumed_from_shot"

    # scenarios: same-size, scale-down (half ranks), scale-up (double ranks)
    declare -A SCENARIOS
    SCENARIOS["same-size"]="$N_RANKS"
    SCENARIOS["scale-down"]=$(( N_RANKS / 2 ))
    SCENARIOS["scale-up"]=$(( N_RANKS * 2 ))

    for scenario in "same-size" "scale-down" "scale-up"; do
        local n_after="${SCENARIOS[$scenario]}"
        local id; id=$(run_id)
        local logfile_before="${LOGS_DIR}/recovery_${scenario}_before_${id}.log"
        local logfile_after="${LOGS_DIR}/recovery_${scenario}_after_${id}.log"
        local cp_dir="${CHECKPOINT_DIR}/recovery_${scenario}_${id}"

        log "recovery: scenario=$scenario ranks_before=$N_RANKS ranks_after=$n_after"

        # 1. Start the job in background with checkpointing
        $MPIRUN --hostfile "$HOSTFILE" -np "$N_RANKS" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            --checkpoint-dir "$cp_dir" \
            --checkpoint-interval 4 \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile_before" &
        local mpi_pid=$!

        # 2. Wait until at least one checkpoint has been flushed
        log "  Waiting for first checkpoint flush..."
        local waited=0
        while ! grep -q 'Flushing checkpoint' "$logfile_before" 2>/dev/null; do
            sleep 5; ((waited += 5))
            [[ "$waited" -ge 300 ]] && { log "ERROR: no checkpoint after 300s"; kill $mpi_pid; exit 1; }
        done
        log "  Checkpoint detected after ${waited}s — injecting failure"

        # 3. Record failure injection time and terminate the node
        local t_inject; t_inject=$(date -u +%s)
        hpcac cluster test-failure \
            --cluster_id "$HPCAC_CLUSTER_ID" \
            --node_private_ip "$FAILURE_NODE_IP"
        wait $mpi_pid || true   # job will exit non-zero

        # 4. Restart on new topology
        log "  Restarting on $n_after ranks..."
        $MPIRUN --hostfile "$HOSTFILE" -np "$n_after" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            --checkpoint-dir "$cp_dir" \
            --checkpoint-interval 4 \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile_after"

        local t_resumed; t_resumed=$(date -u +%s)
        local total_recovery=$(( t_resumed - t_inject ))

        local resumed_from; resumed_from=$(parse_resume_shot "$logfile_after")
        resumed_from="${resumed_from:-N/A}"

        # work_loss = shots since last checkpoint (last checkpoint shot - resumed shot)
        local last_cp_shot
        last_cp_shot=$(grep -oP 'Flushing checkpoint after shot \K[0-9]+' "$logfile_before" | tail -1 || echo "0")
        local work_loss=$(( last_cp_shot > 0 ? resumed_from - last_cp_shot : 0 )) 2>/dev/null || work_loss="N/A"

        csv_append "$csv" \
            "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$scenario,$N_RANKS,$n_after,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$t_inject,$t_resumed,$total_recovery,$work_loss,$resumed_from"

        log "  recovery=${total_recovery}s resumed_from_shot=${resumed_from} work_loss=${work_loss}"
    done

    log "recovery complete → ${csv}"
}

# ─── Mode: elastic ───────────────────────────────────────────────────────────
# Start a clean run, checkpoint at a known shot, then restart on N_RANKS_NEW
# and measure total time-to-solution relative to the uninterrupted baseline.

run_elastic() {
    local csv="${RESULTS_DIR}/elastic_restart.csv"

    csv_header "$csv" \
        "run_id,timestamp,n_ranks_original,n_ranks_new,n_shots,n_timesteps,instance_type,gpu_enabled,shots_at_restart,total_time_s,normalized_time"

    local baseline_time
    baseline_time=$(cat "${RESULTS_DIR}/baseline_time.txt" 2>/dev/null || echo "")

    # Restart topologies relative to N_RANKS: half, same, double
    local new_rank_counts=( $(( N_RANKS / 2 )) "$N_RANKS" $(( N_RANKS * 2 )) )

    for nr in "${new_rank_counts[@]}"; do
        [[ "$nr" -lt 1 ]] && continue
        local id; id=$(run_id)
        local cp_dir="${CHECKPOINT_DIR}/elastic_${id}"
        local logfile_phase1="${LOGS_DIR}/elastic_phase1_${id}.log"
        local logfile_phase2="${LOGS_DIR}/elastic_phase2_r${nr}_${id}.log"

        log "elastic: original=$N_RANKS new=$nr"

        # Phase 1: run until first checkpoint, then kill
        local t_start; t_start=$(date -u +%s)
        $MPIRUN --hostfile "$HOSTFILE" -np "$N_RANKS" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            --checkpoint-dir "$cp_dir" \
            --checkpoint-interval 4 \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile_phase1" &
        local mpi_pid=$!

        # Wait for first flush, then send SIGUSR1 to trigger emergency checkpoint
        local waited=0
        while ! grep -q 'Flushing checkpoint' "$logfile_phase1" 2>/dev/null; do
            sleep 5; ((waited += 5))
            [[ "$waited" -ge 300 ]] && { log "ERROR: no checkpoint after 300s"; kill $mpi_pid; exit 1; }
        done
        pkill -USR1 -x starfwi-fwi || true
        wait $mpi_pid || true

        local shots_at_restart
        shots_at_restart=$(grep -oP 'emergency checkpoint at shot \K[0-9]+' "$logfile_phase1" | tail -1 || \
                           grep -oP 'Flushing checkpoint after shot \K[0-9]+'  "$logfile_phase1" | tail -1 || echo "N/A")

        # Phase 2: restart on new topology; measure remaining time
        local t_restart; t_restart=$(date -u +%s)
        $MPIRUN --hostfile "$HOSTFILE" -np "$nr" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            --checkpoint-dir "$cp_dir" \
            --checkpoint-interval 4 \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile_phase2"
        local t_end; t_end=$(date -u +%s)

        local total_s=$(( t_end - t_start ))
        local norm="N/A"
        [[ -n "$baseline_time" ]] && \
            norm=$(echo "scale=3; $total_s / $baseline_time" | bc)

        csv_append "$csv" \
            "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$N_RANKS,$nr,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$GPU_ENABLED,$shots_at_restart,$total_s,$norm"

        log "  new_ranks=$nr total=${total_s}s normalized=${norm}"
    done

    log "elastic complete → ${csv}"
}

# ─── Mode: cost-perf ─────────────────────────────────────────────────────────
# Run three elasticity policy configurations; collect time-to-solution and
# estimate AWS cost from EC2 instance-hours × current spot/on-demand price.
#
# Policies (defined by the YAML passed to HPC@Cloud):
#   aggressive  — all replacement nodes are spot
#   balanced    — mixed spot + on-demand
#   conservative — all replacements are on-demand
#
# Note: this mode assumes the cluster is already configured with the
# corresponding elasticity policy. Use hpcac cluster create with the
# appropriate YAML before running each policy variant.

run_cost_perf() {
    local csv="${RESULTS_DIR}/cost_perf.csv"

    csv_header "$csv" \
        "run_id,timestamp,policy,n_nodes,n_ranks,n_shots,n_timesteps,instance_type,n_interruptions,time_to_solution_s,spot_price_per_hour,estimated_cost_usd"

    local spot_price
    spot_price=$(query_spot_price)
    log "Current spot price for $INSTANCE_TYPE: \$${spot_price}/hr"

    # Also record on-demand baseline (no interruptions, no checkpointing)
    {
        local id; id=$(run_id)
        local logfile="${LOGS_DIR}/cost_perf_baseline_${id}.log"

        log "cost-perf: policy=on-demand-baseline (no interruptions)"
        local t_start; t_start=$(date -u +%s)
        $MPIRUN --hostfile "$HOSTFILE" -np "$N_RANKS" "$BINARY" \
            --shots "$N_SHOTS" \
            --observed-dir "$OBSERVED_DIR" \
            "$SEGY" "$N_TIMESTEPS" \
            2>&1 | tee "$logfile"
        local t_end; t_end=$(date -u +%s)
        local elapsed=$(( t_end - t_start ))
        local cost; cost=$(estimate_cost "$N_NODES" "$elapsed" "N/A")  # on-demand price TBD

        csv_append "$csv" \
            "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),on-demand-baseline,$N_NODES,$N_RANKS,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,0,$elapsed,N/A,$cost"
        log "  policy=on-demand-baseline elapsed=${elapsed}s"
    }

    # Policy runs — each is repeated REPEAT times for error bars
    local REPEAT="${REPEAT:-3}"
    local policies=("aggressive" "balanced" "conservative")

    for policy in "${policies[@]}"; do
        for (( rep=1; rep<=REPEAT; rep++ )); do
            local id; id=$(run_id)
            local cp_dir="${CHECKPOINT_DIR}/cost_perf_${policy}_r${rep}_${id}"
            local logfile="${LOGS_DIR}/cost_perf_${policy}_rep${rep}_${id}.log"

            log "cost-perf: policy=$policy rep=$rep/$REPEAT"

            launch_watchdogs

            local t_start; t_start=$(date -u +%s)
            $MPIRUN --hostfile "$HOSTFILE" -np "$N_RANKS" "$BINARY" \
                --shots "$N_SHOTS" \
                --observed-dir "$OBSERVED_DIR" \
                --checkpoint-dir "$cp_dir" \
                --checkpoint-interval 4 \
                "$SEGY" "$N_TIMESTEPS" \
                2>&1 | tee "$logfile" || true   # non-zero exit expected on interruption
            # HPC@Cloud handles node replacement and relaunches; wait for the
            # restarted run to finish (reuse same logfile via append)
            # TODO: integrate with hpcac to detect when the restarted job completes
            local t_end; t_end=$(date -u +%s)

            local elapsed=$(( t_end - t_start ))
            local n_interruptions
            n_interruptions=$(grep -c 'emergency checkpoint' "$logfile" || echo 0)
            local cost; cost=$(estimate_cost "$N_NODES" "$elapsed" "$spot_price")

            csv_append "$csv" \
                "$id,$(date -u +%Y-%m-%dT%H:%M:%SZ),$policy,$N_NODES,$N_RANKS,$N_SHOTS,$N_TIMESTEPS,$INSTANCE_TYPE,$n_interruptions,$elapsed,$spot_price,$cost"

            log "  policy=$policy rep=$rep elapsed=${elapsed}s interruptions=$n_interruptions cost=\$${cost}"
        done
    done

    log "cost-perf complete → ${csv}"
}

# ─── Dispatch ────────────────────────────────────────────────────────────────

case "$MODE" in
    baseline)        run_baseline        ;;
    overhead-sweep)  run_overhead_sweep  ;;
    flush-scaling)   run_flush_scaling   ;;
    recovery)        run_recovery        ;;
    elastic)         run_elastic         ;;
    cost-perf)       run_cost_perf       ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage
        ;;
esac
