#include "codelets/forward_propagation.hpp"
#include "acoustics/finite_difference_solver.hpp"
#include "acoustics/receiver_recorder.hpp"
#include "utils/snapshot_writer.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>

namespace starfwi {

/**
 * @brief StarPU codelet definition for forward wave propagation tasks
 *
 * This codelet defines how StarPU should execute forward propagation tasks.
 * Each shot simulation is encapsulated as a StarPU task using this codelet.
 *
 * Configuration:
 *   - CPU function: forward_propagation_cpu
 *   - Number of buffers: 3
 *     - Buffer 0: Velocity model (VECTOR, read-only, STARPU_R)
 *     - Buffer 1: Shot data (VARIABLE, read-write, STARPU_RW)
 *     - Buffer 2: Task config (VARIABLE, read-only, STARPU_R)
 *   - Task name: "forward_propagation"
 *
 * The codelet enables StarPU to:
 *   - Schedule tasks across available workers
 *   - Manage data dependencies automatically
 *   - Handle data transfers in distributed environments
 */
struct starpu_codelet forward_propagation_codelet = {
    .cpu_funcs      = {forward_propagation_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs     = {forward_propagation_cuda},
    .cuda_flags     = {STARPU_CUDA_ASYNC},
#endif
    .cpu_funcs_name = {"forward_propagation_cpu"},
    .nbuffers = 6, // velocity, shot, config, receiver_x, receiver_y, receiver_z
    .modes    = {STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
    // Keep shot (buf 1) and task_config (buf 2) in host RAM even on CUDA workers.
    // ShotData / TaskConfig contain std::vector members with heap-allocated data;
    // StarPU cannot meaningfully serialize them to GPU memory. STARPU_SPECIFIC_NODE_CPU
    // forces these buffers to stay in CPU-accessible memory so STARPU_VARIABLE_GET_PTR
    // returns a valid host pointer from both CPU and CUDA codelet functions.
    .specific_nodes = 1,
    .nodes    = {STARPU_SPECIFIC_NODE_LOCAL,  // buf 0: velocity → GPU for CUDA
                 STARPU_SPECIFIC_NODE_CPU,    // buf 1: shot     → always host RAM
                 STARPU_SPECIFIC_NODE_CPU,    // buf 2: config   → always host RAM
                 STARPU_SPECIFIC_NODE_LOCAL,  // buf 3: recv_x   → GPU for CUDA
                 STARPU_SPECIFIC_NODE_LOCAL,  // buf 4: recv_y   → GPU for CUDA
                 STARPU_SPECIFIC_NODE_LOCAL}, // buf 5: recv_z   → GPU for CUDA
    .name     = "forward_propagation",
    .color    = 0xFF0000};

/**
 * @brief CPU implementation of forward wave propagation for a single shot
 *
 * Pure computational kernel called by StarPU to execute forward modeling.
 * This function performs the complete time-stepping simulation:
 *   1. Creates an acoustic solver instance
 *   2. Initializes wavefields and allocates memory
 *   3. Sets the source position for this specific shot
 *   4. Time-steps through the simulation applying the source wavelet
 *   5. Records receiver data at each timestep
 *   6. Stores the final seismogram data back to the shot structure
 *
 * This kernel is MPI-agnostic - StarPU handles all data movement and
 * scheduling transparently. Each shot is independent, enabling perfect
 * embarrassingly parallel scaling across workers and nodes.
 *
 * @param buffers Array of StarPU data handles:
 *                - buffers[0]: Velocity model (VECTOR of floats, read-only)
 *                - buffers[1]: Shot data structure (VARIABLE ShotData,
 * read-write)
 *                - buffers[2]: Task configuration (VARIABLE TaskConfig,
 * read-only)
 *                - buffers[3]: Receiver X coordinates (VECTOR of floats,
 * read-only)
 *                - buffers[4]: Receiver Y coordinates (VECTOR of floats,
 * read-only)
 *                - buffers[5]: Receiver Z coordinates (VECTOR of floats,
 * read-only)
 * @param cl_arg  Codelet argument containing rank, hostname, verbose flag
 *
 * Note: This function is executed asynchronously by StarPU workers and may
 *       run on any available worker, potentially on different nodes in an
 *       MPI environment.
 */
void forward_propagation_cpu(void *buffers[], void *cl_arg) {
  // Extract data from StarPU buffers
  unsigned int n_velocity = STARPU_VECTOR_GET_NX(buffers[0]);
  float *velocity_data = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  ShotData *shot = (ShotData *)STARPU_VARIABLE_GET_PTR(buffers[1]);
  TaskConfig *task_config = (TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[2]);

  // Extract global receiver arrays
  float *receiver_x = (float *)STARPU_VECTOR_GET_PTR(buffers[3]);
  float *receiver_y = (float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  float *receiver_z = (float *)STARPU_VECTOR_GET_PTR(buffers[5]);

  // Extract rank and hostname from codelet argument
  CodeletArg *arg = (CodeletArg *)cl_arg;
  int rank = arg ? arg->rank : -1;
  const char *hostname = arg ? arg->hostname : "unknown";
  bool verbose = arg ? arg->verbose : false;

  auto start_time = std::chrono::high_resolution_clock::now();

  if (verbose) {
    std::println("[starfwi][{}][forward_propagation_cpu] Processing forward "
                 "propagation step for shot {} at position ({}, {}, {})...",
                 hostname, shot->shot_id, shot->source_x, shot->source_y,
                 shot->source_z);
  }

  // Create SimulationConfig from TaskConfig for FiniteDifferenceSolver
  SimulationConfig config;
  config.grid.nx = task_config->nx;
  config.grid.ny = task_config->ny;
  config.grid.nz = task_config->nz;
  config.grid.dx = task_config->dx;
  config.grid.dy = task_config->dy;
  config.grid.dz = task_config->dz;
  config.time.dt = task_config->dt;
  config.time.nt = task_config->nt;
  config.velocity_model.data.assign(
      velocity_data,
      velocity_data + (task_config->nx * task_config->ny * task_config->nz));

  // Create wave propagator for this shot
  FiniteDifferenceSolver propagator(config);
  if (!propagator.initialize()) {
    std::println(stderr,
                 "[starfwi][{}][forward_propagation_cpu] ERROR: Failed to "
                 "initialize propagator for shot {}",
                 hostname, shot->shot_id);
    return; // Abort this shot
  }

  // Set source position for this specific shot
  propagator.set_source_position(shot->source_x, shot->source_y,
                                 shot->source_z);

  // Initialize receiver recorder with global receiver array
  ReceiverRecorder *recorder = nullptr;
  if (task_config->n_receivers > 0) {
    recorder = new ReceiverRecorder(
        receiver_x, receiver_y, receiver_z, task_config->n_receivers,
        task_config->nt, task_config->nx, task_config->ny, task_config->nz,
        task_config->dx, task_config->dy, task_config->dz);
  }

  const size_t grid_size = task_config->nx * task_config->ny * task_config->nz;

  // Auto-select snapshot storage strategy.
  // NONE (wavefield_storage==2) is always respected — the modeling path never
  // needs snapshots. Otherwise, we prefer host RAM and only fall back to disk
  // when free memory is insufficient (< 70% of currently free physical RAM).
  int actual_storage = -1; // -1 = NONE
  if (task_config->wavefield_storage != 2) {
    const size_t snap_bytes = task_config->nt * grid_size * sizeof(float);
    // Use MemAvailable (not MemFree) so reclaimable page cache is counted.
    // sysinfo().freeram only returns truly free pages and is misleadingly small
    // on machines with large RAM where the kernel aggressively caches file I/O.
    size_t free_ram = 0;
    if (std::ifstream meminfo("/proc/meminfo"); meminfo) {
      std::string line;
      while (std::getline(meminfo, line)) {
        if (line.starts_with("MemAvailable:")) {
          size_t kb = 0;
          std::sscanf(line.c_str(), "MemAvailable: %zu kB", &kb);
          free_ram = kb * 1024ULL;
          break;
        }
      }
    }

    const size_t bytes_per_snap = grid_size * sizeof(float);
    // Divide budget by shots_per_rank: all shots' pressure_snapshots are alive
    // simultaneously during the backward pass, so the total must fit in 70% of RAM.
    const size_t n_shots = std::max(size_t(1), task_config->shots_per_rank);
    const size_t ram_budget = free_ram * 7 / 10 / n_shots;

    if (snap_bytes <= ram_budget) {
      actual_storage = 0; // MEMORY — all snapshots fit
      if (verbose)
        std::println("[starfwi][{}][forward_propagation_cpu] Shot {}: snapshots "
                     "in host RAM ({} MB needed, {} MB available)",
                     hostname, shot->shot_id, snap_bytes >> 20, ram_budget >> 20);
    } else if (bytes_per_snap > ram_budget) {
      actual_storage = 1; // DISK — not even one snapshot fits
      std::println(stderr,
                   "[starfwi][{}][forward_propagation_cpu] Shot {}: insufficient "
                   "host RAM for snapshots ({} MB needed, {} MB available) — using disk",
                   hostname, shot->shot_id, snap_bytes >> 20, ram_budget >> 20);
    } else {
      actual_storage = 2; // HYBRID — keep as many snapshots in RAM as fit
      shot->snapshots_in_ram = ram_budget / bytes_per_snap;
      std::println(stderr,
                   "[starfwi][{}][forward_propagation_cpu] Shot {}: insufficient "
                   "host RAM for all snapshots ({} MB needed, {} MB available) — "
                   "using hybrid ({} of {} snapshots in RAM, rest on disk)",
                   hostname, shot->shot_id, snap_bytes >> 20, ram_budget >> 20,
                   shot->snapshots_in_ram, task_config->nt);
    }
  }
  shot->wavefield_storage_actual = actual_storage;

  const bool use_memory = (actual_storage == 0);
  const bool use_disk   = (actual_storage == 1);
  const bool use_hybrid = (actual_storage == 2);
  const size_t n_in_ram = shot->snapshots_in_ram;

  // Pre-allocate snapshot storage
  std::ofstream wf_out;
  if (use_memory) {
    shot->pressure_snapshots.resize(task_config->nt * grid_size);
  } else if (use_hybrid) {
    shot->pressure_snapshots.resize(n_in_ram * grid_size);
  }
  if (use_disk || use_hybrid) {
    std::string wf_file = std::format("{}/fwd_shot_{}.bin",
                                       task_config->wavefield_dir, shot->shot_id);
    std::filesystem::create_directories(task_config->wavefield_dir);
    wf_out.open(wf_file, std::ios::binary | std::ios::trunc);
    if (!wf_out)
      std::println(stderr,
                   "[starfwi][{}][forward_propagation_cpu] WARNING: cannot open "
                   "wavefield file '{}' — backward propagation will fail",
                   hostname, wf_file);
  }

  // Calculate progress reporting interval (10% of total timesteps)
  size_t progress_interval = task_config->nt / 10;
  if (progress_interval == 0) {
    progress_interval = 1; // Ensure at least 1 to avoid division by zero
  }

  long long disk_write_ms = 0;

  // Time-stepping loop
  for (size_t t = 0; t < task_config->nt; ++t) {
    // Apply source
    if (t < shot->source_wavelet.size()) {
      propagator.apply_source(shot->source_wavelet[t], shot->shot_id);
    }

    // Propagate one time step
    propagator.step();

    // Record at receivers if recorder is initialized
    if (recorder) {
      recorder->record_timestep(propagator, t);
    }

    // Save forward pressure snapshot for adjoint (backward) propagation
    const std::vector<float> &p = propagator.get_pressure_field();
    if (use_memory || (use_hybrid && t < n_in_ram)) {
      std::copy(p.begin(), p.end(),
                shot->pressure_snapshots.begin() + t * grid_size);
    } else if (wf_out) {
      auto t0 = std::chrono::high_resolution_clock::now();
      wf_out.write(reinterpret_cast<const char *>(p.data()),
                   grid_size * sizeof(float));
      disk_write_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }

    // Save wavefield snapshot if enabled
    if (task_config->snapshot_interval > 0 &&
        t % task_config->snapshot_interval == 0) {
      std::string snapshot_file = utils::SnapshotWriter::generate_filename(
          task_config->snapshot_dir, utils::FieldType::PRESSURE, shot->shot_id,
          t);

      auto result = utils::SnapshotWriter::write_snapshot(
          snapshot_file, propagator.get_pressure_field(), task_config->nx,
          task_config->ny, task_config->nz, task_config->dx, task_config->dy,
          task_config->dz, t, task_config->dt, utils::FieldType::PRESSURE);

      if (!result && verbose) {
        std::println(stderr,
                     "[starfwi][{}][forward_propagation_cpu] WARNING: Failed "
                     "to save snapshot at t={}: {}",
                     hostname, t, result.error());
      }
    }

    // Progress reporting (every 10% of total timesteps)
    if (verbose && t % progress_interval == 0) {
      std::println(
          "[starfwi][{}][forward_propagation_cpu] Shot {} progress: {}/{} "
          "({}%)",
          hostname, shot->shot_id, t, task_config->nt,
          (t * 100) / task_config->nt);
    }
  }

  // Copy recorded data back to shot structure
  if (recorder) {
    shot->synthetic_data = recorder->get_synthetic_data();
    delete recorder;
  }

  if (wf_out)
    wf_out.close();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  if (verbose) {
    std::println("[starfwi][{}][forward_propagation_cpu] Shot {} progress: "
                 "100/100. Completed in {:.3f} seconds",
                 hostname, shot->shot_id, total_ms / 1000.0);
  }

  const std::string storage_str = use_memory  ? "ram"
                                 : use_hybrid ? std::format("hybrid({}ram+{}disk)",
                                                    n_in_ram, task_config->nt - n_in_ram)
                                              : "disk";
  std::println("[METRIC] shot={} fwd_total_ms={} fwd_compute_ms={} "
               "fwd_disk_write_ms={} storage={}",
               shot->shot_id, total_ms, total_ms - disk_write_ms,
               disk_write_ms, storage_str);
}

} // namespace starfwi
