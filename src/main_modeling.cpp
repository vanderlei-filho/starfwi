// starfwi-modeling: forward-propagates the true velocity model and writes one
// observed seismogram file per shot to --observed-dir.  These files are then
// consumed by starfwi-fwi as the target data for inversion.

#include "codelets/forward_propagation.hpp"
#include "codelets/save_observed.hpp"
#include "utils/cli_parser.hpp"
#include "utils/receiver_geometry.hpp"
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
    std::println(stderr,
                 "[starfwi-modeling] ERROR: Failed to initialize StarPU-MPI");
    return 1;
  }

  int rank, size;
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
    starfwi::utils::print_ascii_art_title();

  char node_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(node_name, &name_len);

  std::println("[starfwi-modeling][{}] Initialized with {} CPU workers, "
               "{} CUDA workers ({} total workers)",
               node_name, starpu_cpu_worker_get_count(),
               starpu_cuda_worker_get_count(), starpu_worker_get_count());

  std::vector<char> all_processor_names(size * MPI_MAX_PROCESSOR_NAME);
  MPI_Allgather(node_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                all_processor_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                MPI_COMM_WORLD);

  // ========== STEP 2: PARSE COMMAND-LINE ARGUMENTS ==========
  auto parse_result = starfwi::utils::parse_command_line(argc, argv, rank);
  if (!parse_result) {
    if (rank == 0) {
      std::println(stderr, "[starfwi-modeling] Error: {}",
                   parse_result.error());
      starfwi::utils::print_usage(argv[0]);
    }
    starpu_mpi_shutdown();
    return 1;
  }
  starfwi::utils::CliArgs args = *parse_result;

  // ========== STEP 2b: BROADCAST SHARED PARAMETERS FROM RANK 0 ==========
  size_t snapshot_interval_local = args.snapshot_interval;
  MPI_Bcast(&snapshot_interval_local, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  args.snapshot_interval = snapshot_interval_local;

  int n_timesteps_local = args.num_timesteps;
  MPI_Bcast(&n_timesteps_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
  args.num_timesteps = n_timesteps_local;

  // ========== STEP 3: LOAD SEISMIC DATA FROM SEG-Y ==========
  starfwi::SEGYLoader loader;
  auto load_result = loader.load_segy_model(args.segy_filepath, args.verbose);
  if (!load_result) {
    std::println(stderr, "[starfwi-modeling][{}] Error: {}", node_name,
                 load_result.error());
    starpu_mpi_shutdown();
    return 1;
  }
  starfwi::SimulationConfig &config = loader.get_config();

  // ========== STEP 4: CONFIGURE SIMULATION PARAMETERS ==========
  if (config.time.dt == 0.0f) {
    float max_velocity = 0.0f;
    for (const auto &vel : config.velocity_model.data)
      if (vel > max_velocity) max_velocity = vel;

    float min_spacing = (config.grid.ny == 1)
        ? std::min(config.grid.dx, config.grid.dz)
        : std::min({config.grid.dx, config.grid.dy, config.grid.dz});
    float cfl_limit = min_spacing / (max_velocity * std::sqrt(config.grid.ny == 1 ? 2.0f : 3.0f));
    config.time.dt = 0.95f * cfl_limit;

    if (rank == 0 && args.verbose)
      std::println("[starfwi-modeling] Auto-tuned dt: {:.6f} s", config.time.dt);
  }

  config.time.nt = (args.num_timesteps > 0) ? args.num_timesteps : 1000;
  if (rank == 0 && args.verbose)
    std::println("[starfwi-modeling] Timesteps: {}", config.time.nt);

  // ========== STEP 5: SETUP DATA ACQUISITION GEOMETRY ==========
  bool is_2d_model = (config.grid.ny == 1);

  int n_shots_local = args.num_shots;
  MPI_Bcast(&n_shots_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
  size_t n_shots = (n_shots_local > 0)   ? static_cast<size_t>(n_shots_local)
                   : (!config.source.x_positions.empty())
                       ? config.source.x_positions.size()
                       : 1;

  if (rank == 0 && args.verbose)
    std::println("[starfwi-modeling] Number of shots: {}", n_shots);

  if (config.source.frequency <= 0.0f)
    config.source.frequency = 200.0f;

  std::vector<float> base_wavelet = starfwi::generate_ricker_wavelet(
      config.source.frequency, config.time.dt, config.time.nt);

  float domain_length_x = config.grid.dx * (config.grid.nx - 1);
  float x_center = domain_length_x / 2.0f;
  float array_length = domain_length_x * 0.8f;
  float shot_spacing = (n_shots > 1) ? array_length / (n_shots - 1) : 0.0f;
  float shot_x_start = x_center - (array_length / 2.0f);

  std::vector<starfwi::ShotData> shots(n_shots);
  for (size_t i = 0; i < n_shots; ++i) {
    shots[i].shot_id = i + 1;
    if (!config.source.x_positions.empty() && i < config.source.x_positions.size()) {
      shots[i].source_x = config.source.x_positions[i];
      shots[i].source_y = config.source.y_positions[i];
      shots[i].source_z = config.source.z_positions[i];
    } else {
      shots[i].source_x = shot_x_start + i * shot_spacing;
      shots[i].source_y = is_2d_model ? 0.0f : config.grid.dy * config.grid.ny / 2.0f;
      shots[i].source_z = 12.5f;
    }
    shots[i].source_wavelet = base_wavelet;
  }

  // ========== STEP 5b: SETUP GLOBAL RECEIVER ARRAY ==========
  starfwi::ReceiverGeometry receivers;
  size_t n_receivers = 100;
  receivers.reserve(n_receivers);

  float receiver_spacing = array_length / (n_receivers - 1);
  float receiver_x_start = x_center - (array_length / 2.0f);
  for (size_t ir = 0; ir < n_receivers; ++ir) {
    float rx = receiver_x_start + ir * receiver_spacing;
    float ry = is_2d_model ? 0.0f : config.grid.dy * config.grid.ny / 2;
    float rz = config.grid.dz * 2;
    receivers.add_receiver(rx, ry, rz);
  }

  if (rank == 0)
    std::println("[starfwi-modeling] Saving observed data to '{}'",
                 args.observed_dir);

  // ========== STEP 6: REGISTER DATA WITH STARPU ==========
  size_t n_velocity = config.grid.nx * config.grid.ny * config.grid.nz;

  starpu_data_handle_t velocity_handle;
  starpu_vector_data_register(&velocity_handle, STARPU_MAIN_RAM,
                              (uintptr_t)config.velocity_model.data.data(),
                              n_velocity, sizeof(float));
  starpu_mpi_data_register(velocity_handle, 0, 0);

  starpu_data_handle_t receiver_x_handle, receiver_y_handle, receiver_z_handle;
  starpu_vector_data_register(&receiver_x_handle, STARPU_MAIN_RAM,
                              (uintptr_t)receivers.x.data(), n_receivers, sizeof(float));
  starpu_vector_data_register(&receiver_y_handle, STARPU_MAIN_RAM,
                              (uintptr_t)receivers.y.data(), n_receivers, sizeof(float));
  starpu_vector_data_register(&receiver_z_handle, STARPU_MAIN_RAM,
                              (uintptr_t)receivers.z.data(), n_receivers, sizeof(float));
  starpu_mpi_data_register(receiver_x_handle, 10001, 0);
  starpu_mpi_data_register(receiver_y_handle, 10002, 0);
  starpu_mpi_data_register(receiver_z_handle, 10003, 0);

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
  std::strncpy(task_config.observed_dir, args.observed_dir.c_str(), 255);
  task_config.observed_dir[255] = '\0';
  task_config.wavefield_storage = 2; // NONE — modeling never runs backward propagation
  task_config.wavefield_dir[0] = '\0';

  starpu_data_handle_t config_handle;
  starpu_variable_data_register(&config_handle, STARPU_MAIN_RAM,
                                (uintptr_t)&task_config, sizeof(starfwi::TaskConfig));
  starpu_mpi_data_register(config_handle, 999, 0);

  std::vector<starpu_data_handle_t> shot_handles(n_shots);
  for (size_t i = 0; i < n_shots; ++i) {
    starpu_variable_data_register(&shot_handles[i], STARPU_MAIN_RAM,
                                  (uintptr_t)&shots[i], sizeof(starfwi::ShotData));
    starpu_mpi_data_register(shot_handles[i], i + 1, i % size);
  }

  // ========== STEP 7: SUBMIT STARPU TASKS ==========
  starpu_mpi_barrier(MPI_COMM_WORLD);
  double t_start = starpu_timing_now();

  for (size_t i = 0; i < n_shots; ++i) {
    int owner_rank = i % size;

    auto make_arg = [&]() -> starfwi::CodeletArg * {
      auto *arg = new starfwi::CodeletArg();
      std::memset(arg, 0, sizeof(starfwi::CodeletArg));
      arg->rank = owner_rank;
      std::strncpy(arg->hostname,
                   &all_processor_names[owner_rank * MPI_MAX_PROCESSOR_NAME],
                   MPI_MAX_PROCESSOR_NAME - 1);
      arg->verbose = args.verbose;
      return arg;
    };

    // Task 1: Forward propagation
    ret = starpu_mpi_task_insert(
        MPI_COMM_WORLD, &starfwi::forward_propagation_codelet,
        STARPU_R, velocity_handle, STARPU_RW, shot_handles[i],
        STARPU_R, config_handle, STARPU_R, receiver_x_handle,
        STARPU_R, receiver_y_handle, STARPU_R, receiver_z_handle,
        STARPU_CL_ARGS, make_arg(), sizeof(starfwi::CodeletArg), 0);
    if (ret != 0)
      std::println(stderr,
                   "[starfwi-modeling][{}] ERROR: forward_propagation shot {} ({})",
                   node_name, i, ret);

    // Task 2: Save synthetic seismogram as observed data
    ret = starpu_mpi_task_insert(
        MPI_COMM_WORLD, &starfwi::save_observed_codelet,
        STARPU_RW, shot_handles[i], STARPU_R, config_handle,
        STARPU_CL_ARGS, make_arg(), sizeof(starfwi::CodeletArg), 0);
    if (ret != 0)
      std::println(stderr,
                   "[starfwi-modeling][{}] ERROR: save_observed shot {} ({})",
                   node_name, i, ret);
  }

  // ========== STEP 8: WAIT FOR COMPLETION ==========
  std::println("[starfwi-modeling][{}] Waiting for tasks...", node_name);
  starpu_task_wait_for_all();
  std::println("[starfwi-modeling][{}] Done.", node_name);

  // ========== STEP 9: CLEANUP ==========
  for (auto &h : shot_handles) starpu_data_unregister(h);
  starpu_data_unregister(velocity_handle);
  starpu_data_unregister(config_handle);
  starpu_data_unregister(receiver_x_handle);
  starpu_data_unregister(receiver_y_handle);
  starpu_data_unregister(receiver_z_handle);

  starpu_mpi_barrier(MPI_COMM_WORLD);
  double t_end = starpu_timing_now();
  double elapsed_us = t_end - t_start;
  double max_us;
  MPI_Reduce(&elapsed_us, &max_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::println("\n====================================");
    std::println("Modeling complete — observed data written to '{}'",
                 args.observed_dir);
    std::println("Total time: {:.3f} s", max_us / 1e6);
    std::println("====================================");
  }

  starpu_mpi_shutdown();
  return 0;
}
