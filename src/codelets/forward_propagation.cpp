#include "codelets/forward_propagation.hpp"
#include <chrono>
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
    .cpu_funcs = {forward_propagation_cpu},
    .cpu_funcs_name = {"forward_propagation_cpu"},
    .nbuffers =
        3, // velocity model (read) + shot data (read-write) + config (read)
    .modes = {STARPU_R, STARPU_RW, STARPU_R},
    .name = "forward_propagation",
    .color = 0xFF0000};

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
 * @param cl_arg  Codelet argument (unused in current implementation)
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

  // Calculate progress reporting interval (10% of total timesteps)
  size_t progress_interval = task_config->nt / 10;
  if (progress_interval == 0) {
    progress_interval = 1; // Ensure at least 1 to avoid division by zero
  }

  // Time-stepping loop
  for (size_t t = 0; t < task_config->nt; ++t) {
    // Apply source
    if (t < shot->source_wavelet.size()) {
      propagator.apply_source(shot->source_wavelet[t], shot->shot_id);
    }

    // Propagate one time step
    propagator.step();

    // Progress reporting (every 10% of total timesteps)
    if (verbose && t % progress_interval == 0) {
      std::println(
          "[starfwi][{}][forward_propagation_cpu] Shot {} progress: {}/{} "
          "({}%)",
          hostname, shot->shot_id, t, task_config->nt,
          (t * 100) / task_config->nt);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  if (verbose) {
    std::println("[starfwi][{}][forward_propagation_cpu] Shot {} completed in "
                 "{} seconds",
                 hostname, shot->shot_id, duration.count() / 1000.0);
  }
}

} // namespace starfwi
