#include "codelets/forward_propagation.hpp"
#include "utils/cli_parser.hpp"
#include "utils/segy_loader.hpp"
#include "utils/snapshot_writer.hpp"
#include "utils/wavelet.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <print>
#include <starpu.h>
#include <starpu_mpi.h>

int main(int argc, char **argv) {
  // ========== STEP 1: STARPU-MPI INITIALIZATION ==========
  int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
  if (ret != 0) {
    std::println(stderr, "[starfwi] ERROR: Failed to initialize StarPU-MPI");
    return 1;
  }

  int rank, size;
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    starfwi::utils::print_ascii_art_title();
  }

  char node_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(node_name, &name_len);

  int num_cpu_workers = starpu_cpu_worker_get_count();
  int num_cuda_workers = starpu_cuda_worker_get_count();
  int total_workers = starpu_worker_get_count();

  std::println("[starfwi][{}] Initialized with {} CPU workers, {} CUDA workers "
               "({} total workers)",
               node_name, num_cpu_workers, num_cuda_workers, total_workers);

  std::vector<char> all_processor_names(size * MPI_MAX_PROCESSOR_NAME);
  MPI_Allgather(node_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                all_processor_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                MPI_COMM_WORLD);

  // ========== STEP 2: PARSE COMMAND-LINE ARGUMENTS ==========
  auto parse_result = starfwi::utils::parse_command_line(argc, argv, rank);
  if (!parse_result) {
    if (rank == 0) {
      std::println(stderr, "[starfwi] Error: {}", parse_result.error());
      std::println(stderr, "");
      starfwi::utils::print_usage(argv[0]);
    }
    starpu_mpi_shutdown();
    return 1;
  }
  starfwi::utils::CliArgs args = *parse_result;

  // ========== STEP 3: LOAD SEISMIC DATA FROM SEG-Y MODELS ==========
  // All ranks read the same input file (file should be accessible to all nodes)
  starfwi::SEGYLoader loader;
  auto load_result = loader.load_segy_model(args.segy_filepath, args.verbose);
  if (!load_result) {
    std::println(stderr, "[starfwi][{}] Error: {}", node_name,
                 load_result.error());
    starpu_mpi_shutdown();
    return 1;
  }
  starfwi::SimulationConfig &config = loader.get_config();

  // ========== STEP 4: CONFIGURE SIMULATION PARAMETERS ==========
  // Set timestep (dt) based on CFL stability condition
  if (config.time.dt == 0.0f) {
    // CFL condition for 3D acoustic wave equation: dt < dx / (v_max * sqrt(3))
    // Find maximum velocity in the model
    float max_velocity = 0.0f;
    for (const auto &vel : config.velocity_model.data) {
      if (vel > max_velocity) {
        max_velocity = vel;
      }
    }

    // Find minimum grid spacing
    float min_spacing = config.grid.dx;
    if (config.grid.ny > 1 && config.grid.dy > 0.0f) {
      min_spacing = std::min(min_spacing, config.grid.dy);
    }
    min_spacing = std::min(min_spacing, config.grid.dz);

    // Calculate stable timestep with safety factor of 0.95
    float cfl_limit =
        min_spacing /
        (max_velocity * std::sqrt(config.grid.ny == 1 ? 2.0f : 3.0f));
    config.time.dt = 0.95f * cfl_limit;

    if (rank == 0 and args.verbose) {
      std::println(
          "[starfwi] Using auto-tuned timedelta: {:.6f} seconds (0.5 × CFL "
          "stability limit)",
          config.time.dt);
    }
  }

  // Override timesteps if specified via command line
  if (args.num_iterations > 0) {
    config.time.nt = args.num_iterations;
    if (rank == 0 and args.verbose) {
      std::println("[starfwi] Timesteps (from command-line): {}",
                   args.num_iterations);
    }
  } else {
    config.time.nt = 1000;
    if (rank == 0 and args.verbose) {
      std::println("[starfwi] Timesteps (using default): 1000");
    }
  }

  // ========== STEP 5: SETUP DATA ACQUISITION GEOMETRY ==========
  // Set up airgun sources
  // TODO: make this configurable

  // Detect model dimensionality for optimization
  bool is_2d_model = (config.grid.ny == 1);

  // Determine number of shots
  // TODO: make this configurable
  size_t n_shots;
  if (!config.source.x_positions.empty()) {
    n_shots = config.source.x_positions.size();
  } else {
    n_shots = 1; // Default: 8 shots
  }

  // All ranks allocate shots vector
  std::vector<starfwi::ShotData> shots(n_shots);

  // All ranks generate shot data for shots they own (round-robin distribution)
  // TODO: make source frequency configurable, and improve round-robin
  // distribution
  if (config.source.frequency <= 0.0f) {
    config.source.frequency = 20.0f;
  }
  float source_frequency = config.source.frequency;

  // All ranks generate the same wavelet
  std::vector<float> base_wavelet = starfwi::generate_ricker_wavelet(
      source_frequency, config.time.dt, config.time.nt);

  // ALL ranks initialize ALL shots (required for StarPU-MPI data registration)
  // Only the owning rank will actually use this data initially
  for (size_t i = 0; i < n_shots; ++i) {
    int owner_rank = i % size;

    shots[i].shot_id = i + 1; // Shot numbering starts at 1

    // Set source positions
    if (i < config.source.x_positions.size()) {
      shots[i].source_x = config.source.x_positions[i];
      shots[i].source_y = config.source.y_positions[i];
      shots[i].source_z = config.source.z_positions[i];
    } else {
      // Default: place shots along a line in x-direction
      shots[i].source_x = config.grid.dx * (10 + i * 10);

      // For 2D models, Y position is always 0 (no Y dimension)
      // For 3D models, place at center of Y range
      if (is_2d_model) {
        shots[i].source_y = 0.0f;
      } else {
        shots[i].source_y = config.grid.dy * config.grid.ny / 2;
      }

      shots[i].source_z = config.grid.dz * config.grid.nz / 2;
    }

    // Copy wavelet (all ranks need valid wavelet vector for StarPU
    // registration)
    shots[i].source_wavelet = base_wavelet;
  }

  // Print shot distribution summary for each rank
  std::vector<size_t> my_shots;
  for (size_t i = 0; i < n_shots; ++i) {
    if ((i % size) == rank) {
      my_shots.push_back(i + 1); // Shot IDs start at 1
    }
  }

  // Print each rank's assignment sequentially
  for (int r = 0; r < size; ++r) {
    if (rank == r) {
      if (my_shots.empty()) {
        std::println("[starfwi][{}] Will process 0 shots", node_name);
      } else {
        std::print("[starfwi][{}] Will process {} shot(s): [", node_name,
                   my_shots.size());
        for (size_t i = 0; i < my_shots.size(); ++i) {
          std::print("{}", my_shots[i]);
          if (i < my_shots.size() - 1) {
            std::print(", ");
          }
        }
        std::println("]");
      }
    }
    starpu_mpi_barrier(MPI_COMM_WORLD);
  }

  std::println(
      "[starfwi][{}] Data acquisition geometry initialization completed",
      node_name);

  // Save velocity model snapshot (rank 0 only)
  if (rank == 0 && args.snapshot_interval > 0) {
    std::string vel_file = starfwi::utils::SnapshotWriter::generate_filename(
        args.snapshot_dir, starfwi::utils::FieldType::VELOCITY, 0, 0);
    bool success = starfwi::utils::SnapshotWriter::write_snapshot(
        vel_file, config.velocity_model.data, config.grid.nx, config.grid.ny,
        config.grid.nz, config.grid.dx, config.grid.dy, config.grid.dz, 0,
        config.time.dt, starfwi::utils::FieldType::VELOCITY);
    if (success && args.verbose) {
      std::println("[starfwi] Saved velocity model to {}", vel_file);
    }
  }

  // ========== STEP 6: REGISTER DATA WITH STARPU ==========
  // All ranks have loaded the velocity model, so we can register it directly
  size_t n_velocity = config.grid.nx * config.grid.ny * config.grid.nz;

  // Register velocity model (VECTOR handle)
  starpu_data_handle_t velocity_handle;
  starpu_vector_data_register(&velocity_handle, STARPU_MAIN_RAM,
                              (uintptr_t)config.velocity_model.data.data(),
                              n_velocity, sizeof(float));
  starpu_mpi_data_register(velocity_handle, 0, 0); // Tag 0, owner rank 0

  // Register task configuration (VARIABLE handle)
  // All ranks need to register with the same data structure
  starfwi::TaskConfig task_config;
  task_config.nx = config.grid.nx;
  task_config.ny = config.grid.ny;
  task_config.nz = config.grid.nz;
  task_config.dx = config.grid.dx;
  task_config.dy = config.grid.dy;
  task_config.dz = config.grid.dz;
  task_config.dt = config.time.dt;
  task_config.nt = config.time.nt;
  task_config.snapshot_interval = args.snapshot_interval;
  std::strncpy(task_config.snapshot_dir, args.snapshot_dir.c_str(), 255);
  task_config.snapshot_dir[255] = '\0';

  starpu_data_handle_t config_handle;
  starpu_variable_data_register(&config_handle, STARPU_MAIN_RAM,
                                (uintptr_t)&task_config,
                                sizeof(starfwi::TaskConfig));
  starpu_mpi_data_register(config_handle, 999, 0); // Tag 999, owner rank 0

  // Register shot data (VARIABLE handles, distributed round-robin ownership)
  // Distribution based on array index: rank = index % size
  // Example with 3 ranks: Rank 0 owns shots {1,4,7}, Rank 1 owns {2,5,8}, etc.
  std::vector<starpu_data_handle_t> shot_handles(n_shots);
  for (size_t i = 0; i < n_shots; ++i) {
    starpu_variable_data_register(&shot_handles[i], STARPU_MAIN_RAM,
                                  (uintptr_t)&shots[i],
                                  sizeof(starfwi::ShotData));
    int owner_rank = i % size;
    starpu_mpi_data_register(shot_handles[i], i + 1, owner_rank);
  }
  std::println("[starfwi][{}] Data registration with StarPU completed",
               node_name);

  // Synchronize all ranks before starting timer
  starpu_mpi_barrier(MPI_COMM_WORLD);
  double total_start_time = starpu_timing_now();

  // ========== STEP 7: SUBMIT STARPU TASKS ==========
  // ALL ranks submit ALL tasks (collective submission pattern for StarPU-MPI)
  // StarPU-MPI automatically executes each task on the rank that owns the RW
  // data
  for (size_t i = 0; i < n_shots; ++i) {
    int owner_rank = i % size;

    // Create codelet argument with owner rank and hostname information
    starfwi::CodeletArg *arg = new starfwi::CodeletArg();
    std::memset(arg, 0, sizeof(starfwi::CodeletArg)); // Zero-initialize
    arg->rank = owner_rank;
    std::strncpy(arg->hostname,
                 &all_processor_names[owner_rank * MPI_MAX_PROCESSOR_NAME],
                 MPI_MAX_PROCESSOR_NAME - 1);
    arg->hostname[MPI_MAX_PROCESSOR_NAME - 1] = '\0'; // Ensure null termination
    arg->verbose = args.verbose;

    if (args.verbose && rank == 0 && i == 0) {
      std::println("[DEBUG] Submitting shot {} to rank {} on hostname: '{}', "
                   "verbose={}, "
                   "sizeof(CodeletArg)={}",
                   i, owner_rank, arg->hostname, arg->verbose,
                   sizeof(starfwi::CodeletArg));
    }

    int ret = starpu_mpi_task_insert(
        MPI_COMM_WORLD, &starfwi::forward_propagation_codelet, STARPU_R,
        velocity_handle, // Velocity model (read-only, owned by rank 0)
        STARPU_RW,
        shot_handles[i], // Shot data (read-write, owned by owner_rank)
        STARPU_R,
        config_handle, // Task configuration (read-only, owned by rank 0)
        STARPU_CL_ARGS, arg, sizeof(starfwi::CodeletArg), 0);

    // TODO: improve error handling here
    if (ret != 0) {
      std::println(stderr,
                   "[starfwi][{}] ERROR: submitting MPI task for shot {} "
                   "(error code: {})",
                   node_name, i, ret);
    }
  }

  // ========== STEP 8: WAIT FOR TASKS COMPLETION ==========
  std::println("[starfwi][{}] Waiting for StarPU tasks to complete...",
               node_name);
  auto start_wait = std::chrono::high_resolution_clock::now();

  starpu_task_wait_for_all();

  auto end_wait = std::chrono::high_resolution_clock::now();
  auto wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_wait - start_wait);

  std::println("[starfwi][{}] StarPU tasks completed in {} seconds!", node_name,
               wait_duration.count() / 1000.0);

  // ========== STEP 9: CLEANUP ==========
  std::println("[starfwi][{}] Unregistering StarPU data...", node_name);

  for (auto &handle : shot_handles) {
    starpu_data_unregister(handle);
  }
  starpu_data_unregister(velocity_handle);
  starpu_data_unregister(config_handle);

  std::println("[starfwi][{}] Cleanup completed", node_name);

  // Synchronize all ranks before stopping timer
  starpu_mpi_barrier(MPI_COMM_WORLD);
  double total_end_time = starpu_timing_now();
  double total_duration_us = total_end_time - total_start_time;

  // Get the maximum time across all ranks (slowest rank determines total time)
  double max_time_us;
  MPI_Reduce(&total_duration_us, &max_time_us, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    std::println("\n====================================");
    std::println("StarFWI simulation complete!");
    std::println("Total execution time: {:.3f} seconds",
                 max_time_us / 1000000.0);
    std::println("====================================");
  }

  starpu_mpi_shutdown();
  return 0;
}
