#include "codelets/forward_propagation.hpp"
#include "utils/cli_parser.hpp"
#include "utils/receiver_geometry.hpp"
#include "utils/segy_loader.hpp"
#include "utils/seismogram_io.hpp"
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

  // ========== STEP 2b: BROADCAST SHARED PARAMETERS FROM RANK 0 ==========
  // Broadcast snapshot_interval from rank 0 to ensure consistency
  // (snapshot_dir can remain different per rank for distributed storage)
  size_t snapshot_interval_local = args.snapshot_interval;
  MPI_Bcast(&snapshot_interval_local, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  args.snapshot_interval = snapshot_interval_local;

  // Broadcast num_iterations from rank 0 to all ranks
  int n_iterations_local = args.num_iterations; // -1 if not specified
  MPI_Bcast(&n_iterations_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
  args.num_iterations = n_iterations_local;

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
  // Detect model dimensionality for optimization
  bool is_2d_model = (config.grid.ny == 1);

  // Determine number of shots (rank 0 decides, broadcasts to all)
  int n_shots_local = args.num_shots; // -1 if not specified

  // Broadcast num_shots from rank 0 to all ranks
  MPI_Bcast(&n_shots_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

  size_t n_shots;
  if (n_shots_local > 0) {
    // Use CLI parameter (from rank 0)
    n_shots = static_cast<size_t>(n_shots_local);
  } else if (!config.source.x_positions.empty()) {
    // Use SEG-Y source positions if available
    n_shots = config.source.x_positions.size();
  } else {
    // Default: 1 shot
    n_shots = 1;
  }

  if (rank == 0 && args.verbose) {
    std::println("[starfwi] Number of shots: {}", n_shots);
  }

  // All ranks allocate shots vector
  std::vector<starfwi::ShotData> shots(n_shots);

  // Source wavelet configuration
  if (config.source.frequency <= 0.0f) {
    config.source.frequency = 20.0f;
  }
  float source_frequency = config.source.frequency;

  // All ranks generate the same wavelet
  std::vector<float> base_wavelet = starfwi::generate_ricker_wavelet(
      source_frequency, config.time.dt, config.time.nt);

  // Calculate geometry parameters (used for both shots and receivers)
  float domain_length_x = config.grid.dx * (config.grid.nx - 1);
  float x_center = domain_length_x / 2.0f;
  float array_length = domain_length_x * 0.8f; // Use 80% of domain width

  // Calculate shot placement geometry (centered, symmetric spacing)
  float shot_spacing = (n_shots > 1) ? array_length / (n_shots - 1) : 0.0f;
  float shot_x_start = x_center - (array_length / 2.0f);
  float shot_depth = 40.0f; // 40 meters below surface

  // ALL ranks initialize ALL shots (required for StarPU-MPI data registration)
  for (size_t i = 0; i < n_shots; ++i) {
    shots[i].shot_id = i + 1; // Shot numbering starts at 1

    // Set source positions
    if (!config.source.x_positions.empty() &&
        i < config.source.x_positions.size()) {
      // Use SEG-Y positions if available
      shots[i].source_x = config.source.x_positions[i];
      shots[i].source_y = config.source.y_positions[i];
      shots[i].source_z = config.source.z_positions[i];
    } else {
      // Place shots in centered, evenly-spaced line at 40m depth
      shots[i].source_x = shot_x_start + i * shot_spacing;

      // For 2D models, Y position is always 0
      // For 3D models, place at center of Y range
      if (is_2d_model) {
        shots[i].source_y = 0.0f;
      } else {
        shots[i].source_y = config.grid.dy * config.grid.ny / 2.0f;
      }

      shots[i].source_z = shot_depth;
    }

    // Copy wavelet (all ranks need valid wavelet vector for StarPU
    // registration)
    shots[i].source_wavelet = base_wavelet;

    // Load observed data if available and not generating observed data
    // (FWI mode: load pre-computed observed data from true model)
    if (!args.generate_observed && !args.observed_dir.empty()) {
      std::string observed_file = starfwi::utils::SeismogramIO::generate_filename(
          args.observed_dir, shots[i].shot_id);
      starfwi::utils::SeismogramIO::Header header;
      std::vector<float> observed_data;
      auto load_result = starfwi::utils::SeismogramIO::load(observed_file, observed_data, header);
      if (load_result) {
        shots[i].observed_data = std::move(observed_data);
        if (rank == 0 && args.verbose) {
          std::println("[starfwi] Loaded observed data for shot {} from {}",
                       shots[i].shot_id, observed_file);
        }
      } else if (rank == 0 && args.verbose) {
        std::println("[starfwi] No observed data found for shot {} ({})",
                     shots[i].shot_id, load_result.error());
      }
    }
  }

  // ========== STEP 5b: SETUP GLOBAL RECEIVER ARRAY ==========
  // Receivers are fixed for all shots (standard seismic survey geometry)
  // TODO: make receiver geometry configurable
  starfwi::ReceiverGeometry receivers;
  size_t n_receivers = 100; // Total number of receivers in survey
  receivers.reserve(n_receivers);

  // Distribute 100 receivers evenly along X axis at surface, centered on domain
  // (symmetric spacing - reuses domain_length_x, x_center, array_length from
  // above)
  float receiver_spacing = array_length / (n_receivers - 1);
  float receiver_x_start = x_center - (array_length / 2.0f);

  for (size_t ir = 0; ir < n_receivers; ++ir) {
    float rx = receiver_x_start + ir * receiver_spacing;

    // For 2D models, Y position is always 0
    // For 3D models, place at center of Y range
    float ry;
    if (is_2d_model) {
      ry = 0.0f;
    } else {
      ry = config.grid.dy * config.grid.ny / 2;
    }

    // Place receivers at surface (z = 0) or slightly below
    float rz = config.grid.dz * 2; // 2 grid points below surface

    receivers.add_receiver(rx, ry, rz);
  }

  if (rank == 0 && args.verbose) {
    std::println("[starfwi] Initialized {} global receivers", receivers.size());
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
    auto result = starfwi::utils::SnapshotWriter::write_snapshot(
        vel_file, config.velocity_model.data, config.grid.nx, config.grid.ny,
        config.grid.nz, config.grid.dx, config.grid.dy, config.grid.dz, 0,
        config.time.dt, starfwi::utils::FieldType::VELOCITY);
    if (result) {
      if (args.verbose) {
        std::println("[starfwi] Saved velocity model to {}", vel_file);
      }
    } else {
      std::println(stderr, "[starfwi] ERROR: Failed to save velocity model: {}",
                   result.error());
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

  // Register global receiver arrays (VECTOR handles)
  starpu_data_handle_t receiver_x_handle, receiver_y_handle, receiver_z_handle;
  starpu_vector_data_register(&receiver_x_handle, STARPU_MAIN_RAM,
                              (uintptr_t)receivers.x.data(), n_receivers,
                              sizeof(float));
  starpu_vector_data_register(&receiver_y_handle, STARPU_MAIN_RAM,
                              (uintptr_t)receivers.y.data(), n_receivers,
                              sizeof(float));
  starpu_vector_data_register(&receiver_z_handle, STARPU_MAIN_RAM,
                              (uintptr_t)receivers.z.data(), n_receivers,
                              sizeof(float));
  starpu_mpi_data_register(receiver_x_handle, 100, 0); // Tag 100, owner rank 0
  starpu_mpi_data_register(receiver_y_handle, 101, 0); // Tag 101, owner rank 0
  starpu_mpi_data_register(receiver_z_handle, 102, 0); // Tag 102, owner rank 0

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
  task_config.n_receivers = n_receivers;

  // FWI options
  task_config.generate_observed = args.generate_observed;
  std::strncpy(task_config.observed_dir, args.observed_dir.c_str(), 255);
  task_config.observed_dir[255] = '\0';

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
        velocity_handle,             // Velocity model (read-only, rank 0)
        STARPU_RW, shot_handles[i],  // Shot data (read-write, owner_rank)
        STARPU_R, config_handle,     // Task configuration (read-only, rank 0)
        STARPU_R, receiver_x_handle, // Receiver X coords (read-only, rank 0)
        STARPU_R, receiver_y_handle, // Receiver Y coords (read-only, rank 0)
        STARPU_R, receiver_z_handle, // Receiver Z coords (read-only, rank 0)
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

  std::println("[starfwi][{}] This node StarPU tasks completed in {} seconds!",
               node_name, wait_duration.count() / 1000.0);

  // ========== STEP 9: CLEANUP ==========
  std::println("[starfwi][{}] Unregistering StarPU data from node...",
               node_name);

  for (auto &handle : shot_handles) {
    starpu_data_unregister(handle);
  }
  starpu_data_unregister(velocity_handle);
  starpu_data_unregister(config_handle);
  starpu_data_unregister(receiver_x_handle);
  starpu_data_unregister(receiver_y_handle);
  starpu_data_unregister(receiver_z_handle);

  std::println("[starfwi][{}] Node cleanup completed", node_name);

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
