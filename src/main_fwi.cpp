// starfwi-fwi: loads observed seismograms from --observed-dir, runs the
// FWI inversion pipeline (forward propagation → misfit → backward
// propagation) for each shot using StarPU-MPI.

#include "codelets/backward_propagation.hpp"
#include "codelets/compute_misfit.hpp"
#include "codelets/forward_propagation.hpp"
#include "utils/cli_parser.hpp"
#include "utils/receiver_geometry.hpp"
#include "utils/segy_loader.hpp"
#include "utils/seismogram_io.hpp"
#include "utils/snapshot_writer.hpp"
#include "utils/wavelet.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <print>
#include <starpu.h>
#include <starpu_mpi.h>
#include <starpu_mpi_ft.h>

static int g_cp_comm_size = 1;
static int fwi_shots_backup_rank(int rank) {
  return (rank + 1) % g_cp_comm_size;
}

// Set by SIGUSR1 when the IMDS watchdog detects a spot interruption notice.
// Checked collectively at each shot boundary to trigger an emergency checkpoint.
static std::atomic<bool> g_interrupt_requested{false};
static void handle_sigusr1(int) { g_interrupt_requested.store(true); }

// Returns true if any rank has received a spot interruption signal.
// Must be called collectively by all ranks.
static bool check_spot_interrupt(MPI_Comm comm) {
  int local = g_interrupt_requested.load() ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm);
  return global != 0;
}

// Returns the total size in bytes of all regular files under `dir`.
// Returns 0 if the directory does not exist.
static uintptr_t dir_size_bytes(const std::string &dir) {
  namespace fs = std::filesystem;
  uintptr_t total = 0;
  std::error_code ec;
  for (const auto &entry : fs::recursive_directory_iterator(dir, ec))
    if (entry.is_regular_file(ec))
      total += entry.file_size(ec);
  return total;
}

// Flush a checkpoint, measure wall time and bytes written, and print a
// [METRIC] line on rank 0 so the experiment runner can parse it.
static void flush_checkpoint_with_metrics(starpu_mpi_checkpoint_template_t tmpl,
                                          const std::string &cp_dir,
                                          int rank) {
  uintptr_t size_before = dir_size_bytes(cp_dir);
  auto t0 = std::chrono::steady_clock::now();

  starpu_mpi_checkpoint_flush_to_storage(tmpl);

  auto t1 = std::chrono::steady_clock::now();
  uintptr_t size_after = dir_size_bytes(cp_dir);

  if (rank == 0) {
    long long flush_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double cp_mb = static_cast<double>(size_after - size_before) / (1024.0 * 1024.0);
    std::println("[METRIC] flush_time_ms={} checkpoint_size_mb={:.2f}",
                 flush_ms, cp_mb);
  }
}

int main(int argc, char **argv) {
  // ========== STEP 1: STARPU-MPI INITIALIZATION ==========
  int ret = starpu_mpi_init_conf(&argc, &argv, 1, MPI_COMM_WORLD, NULL);
  if (ret != 0) {
    std::println(stderr, "[starfwi-fwi] ERROR: Failed to initialize StarPU-MPI");
    return 1;
  }
  starpu_mpi_checkpoint_init();
  signal(SIGUSR1, handle_sigusr1);

  int rank, size;
  starpu_mpi_comm_rank(MPI_COMM_WORLD, &rank);
  starpu_mpi_comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
    starfwi::utils::print_ascii_art_title();

  char node_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(node_name, &name_len);

  std::println("[starfwi-fwi][{}] Initialized with {} CPU workers, "
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
      std::println(stderr, "[starfwi-fwi] Error: {}", parse_result.error());
      starfwi::utils::print_usage(argv[0]);
    }
    starpu_mpi_shutdown();
    return 1;
  }
  starfwi::utils::CliArgs args = *parse_result;

  // ========== STEP 2b: CHECKPOINT RESTART DETECTION ==========
  int restart_flag = 0;
  if (!args.checkpoint_dir.empty()) {
    starpu_mpi_init_from_checkpoint(args.checkpoint_dir.c_str(), &restart_flag);
    if (rank == 0) {
      if (restart_flag)
        std::println("[starfwi-fwi] Checkpoint found — resuming "
                     "(cp_id={}, cp_inst={}, old_n_ranks={})",
                     starpu_mpi_checkpoint_get_restart_cp_id(),
                     starpu_mpi_checkpoint_get_restart_cp_inst(),
                     starpu_mpi_checkpoint_get_restart_n_ranks());
      else
        std::println("[starfwi-fwi] No checkpoint found in '{}' — starting fresh",
                     args.checkpoint_dir);
    }
  }

  // ========== STEP 2c: BROADCAST SHARED PARAMETERS FROM RANK 0 ==========
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
    std::println(stderr, "[starfwi-fwi][{}] Error: {}", node_name,
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
      std::println("[starfwi-fwi] Auto-tuned dt: {:.6f} s", config.time.dt);
  }

  config.time.nt = (args.num_timesteps > 0) ? args.num_timesteps : 1000;
  if (rank == 0 && args.verbose)
    std::println("[starfwi-fwi] Timesteps: {}", config.time.nt);

  // ========== STEP 4b: APPLY VELOCITY PERTURBATION ==========
  // Scale the initial model so it differs from the true model used during
  // modeling, producing a non-zero misfit and a meaningful gradient.
  if (args.velocity_scale != 1.0f) {
    for (float &v : config.velocity_model.data)
      v *= args.velocity_scale;
    if (rank == 0)
      std::println("[starfwi-fwi] Applied velocity scale {:.3f} to initial model.",
                   args.velocity_scale);
  }

  // ========== STEP 5: SETUP DATA ACQUISITION GEOMETRY ==========
  bool is_2d_model = (config.grid.ny == 1);

  int n_shots_local = args.num_shots;
  MPI_Bcast(&n_shots_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
  size_t n_shots = (n_shots_local > 0)   ? static_cast<size_t>(n_shots_local)
                   : (!config.source.x_positions.empty())
                       ? config.source.x_positions.size()
                       : 1;

  if (rank == 0 && args.verbose)
    std::println("[starfwi-fwi] Number of shots: {}", n_shots);

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

  // ========== STEP 5c: LOAD OBSERVED DATA FROM DISK ==========
  // All ranks load all shots so the task submission condition is uniformly
  // true across the communicator (StarPU-MPI requires collective submission).
  if (rank == 0)
    std::println("[starfwi-fwi] Loading observed data from '{}'...",
                 args.observed_dir);

  for (size_t i = 0; i < n_shots; ++i) {
    std::string filename = starfwi::utils::SeismogramIO::generate_filename(
        args.observed_dir, shots[i].shot_id);
    starfwi::utils::SeismogramIO::Header hdr;
    std::vector<float> data;
    auto res = starfwi::utils::SeismogramIO::load(filename, data, hdr);
    if (res) {
      shots[i].observed_data = std::move(data);
    } else {
      if (rank == 0)
        std::println(stderr,
                     "[starfwi-fwi] WARNING: could not load observed data for "
                     "shot {} ({})",
                     shots[i].shot_id, res.error());
    }
  }

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
  task_config.observed_dir[0] = '\0'; // not used in FWI binary
  task_config.wavefield_storage =
      (args.wavefield_storage == starfwi::utils::CliArgs::WavefieldStorage::MEMORY) ? 0 : 1;
  std::strncpy(task_config.wavefield_dir, args.wavefield_dir.c_str(), 255);
  task_config.wavefield_dir[255] = '\0';

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

  // ========== STEP 6b: CHECKPOINT TEMPLATE REGISTRATION ==========
  g_cp_comm_size = size;
  starpu_mpi_checkpoint_template_t cp_template = nullptr;
  int shots_completed = 0;

  if (!args.checkpoint_dir.empty()) {
    starpu_mpi_checkpoint_set_storage_path(args.checkpoint_dir.c_str());

    int vel_backup_rank = (size > 1) ? 1 : 0;
    starpu_mpi_checkpoint_template_register(
        &cp_template, /*cp_id=*/1, /*domain=*/0,
        STARPU_R, velocity_handle, vel_backup_rank,
        STARPU_VALUE, &shots_completed, sizeof(int),
            (starpu_mpi_tag_t)1000, fwi_shots_backup_rank,
        0);

    if (restart_flag) {
      int cp_id       = starpu_mpi_checkpoint_get_restart_cp_id();
      int cp_inst     = starpu_mpi_checkpoint_get_restart_cp_inst();
      int old_n_ranks = starpu_mpi_checkpoint_get_restart_n_ranks();

      if (rank == 0)
        starpu_mpi_checkpoint_restore_handle(cp_id, cp_inst, 0, 0, velocity_handle);

      int old_rank = (old_n_ranks > 0) ? (rank % old_n_ranks) : 0;
      int restored = 0;
      if (starpu_mpi_checkpoint_restore_value(cp_id, cp_inst, old_rank,
                                              1000, &restored, sizeof(int)) == 0)
        shots_completed = restored;

      MPI_Allreduce(MPI_IN_PLACE, &shots_completed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (rank == 0)
        std::println("[starfwi-fwi] Resuming from shot {} of {}",
                     shots_completed, n_shots);
    }
  }

  // ========== STEP 7: OUTER FWI ITERATION LOOP ==========
  starpu_mpi_barrier(MPI_COMM_WORLD);
  double t_start = starpu_timing_now();

  std::vector<float> full_gradient(n_velocity, 0.0f);

  bool interrupted = false;
  for (int iter = 0; iter < args.num_iterations && !interrupted; ++iter) {
    if (rank == 0)
      std::println("\n[starfwi-fwi] ===== Iteration {}/{} =====",
                   iter + 1, args.num_iterations);

    // -------- Submit tasks for every shot --------
    size_t shots_at_resume = (size_t)shots_completed;
    for (size_t i = (size_t)shots_completed; i < n_shots && !interrupted; ++i) {
      int owner_rank = i % size;

      // Check for spot interruption notice before submitting the next shot.
      // The check is collective: SIGUSR1 only reaches ranks on the affected
      // node, but the emergency flush must involve all ranks.
      if (cp_template && check_spot_interrupt(MPI_COMM_WORLD)) {
        shots_completed = static_cast<int>(i);
        if (rank == 0)
          std::println("[starfwi-fwi] Spot interruption detected — "
                       "flushing emergency checkpoint at shot {} of {} and exiting",
                       shots_completed, n_shots);
        flush_checkpoint_with_metrics(cp_template, args.checkpoint_dir, rank);
        interrupted = true;
        break;
      }

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
                     "[starfwi-fwi][{}] ERROR: forward_propagation shot {} ({})",
                     node_name, i, ret);

      // Task 2: Compute misfit
      ret = starpu_mpi_task_insert(
          MPI_COMM_WORLD, &starfwi::compute_misfit_codelet,
          STARPU_RW, shot_handles[i], STARPU_R, config_handle,
          STARPU_CL_ARGS, make_arg(), sizeof(starfwi::CodeletArg), 0);
      if (ret != 0)
        std::println(stderr,
                     "[starfwi-fwi][{}] ERROR: compute_misfit shot {} ({})",
                     node_name, i, ret);

      // Task 3: Backward (adjoint) propagation
      ret = starpu_mpi_task_insert(
          MPI_COMM_WORLD, &starfwi::backward_propagation_codelet,
          STARPU_R, velocity_handle, STARPU_RW, shot_handles[i],
          STARPU_R, config_handle, STARPU_R, receiver_x_handle,
          STARPU_R, receiver_y_handle, STARPU_R, receiver_z_handle,
          STARPU_CL_ARGS, make_arg(), sizeof(starfwi::CodeletArg), 0);
      if (ret != 0)
        std::println(stderr,
                     "[starfwi-fwi][{}] ERROR: backward_propagation shot {} ({})",
                     node_name, i, ret);

      // Periodic checkpoint flush
      if (cp_template && args.checkpoint_interval > 0 &&
          ((i + 1 - shots_at_resume) % args.checkpoint_interval == 0)) {
        shots_completed = static_cast<int>(i + 1);
        if (rank == 0)
          std::println("[starfwi-fwi] Flushing checkpoint after shot {} of {}",
                       shots_completed, n_shots);
        flush_checkpoint_with_metrics(cp_template, args.checkpoint_dir, rank);
      }
    }
    // All shots submitted for this iteration — reset counter for next
    if (!interrupted)
      shots_completed = 0;

    if (interrupted) break;

    // -------- Wait for all tasks --------
    if (args.verbose)
      std::println("[starfwi-fwi][{}] Waiting for tasks (iter {})...",
                   node_name, iter + 1);
    starpu_task_wait_for_all();

    // -------- Gradient and misfit reduction --------
    std::vector<float> local_gradient(n_velocity, 0.0f);
    float local_misfit = 0.0f;

    for (size_t i = 0; i < n_shots; ++i) {
      if ((int)(i % size) != rank) continue;
      starpu_data_acquire(shot_handles[i], STARPU_R);
      if (!shots[i].gradient.empty()) {
        for (size_t j = 0; j < n_velocity; ++j)
          local_gradient[j] += shots[i].gradient[j];
      }
      local_misfit += shots[i].misfit;
      starpu_data_release(shot_handles[i]);
    }

    MPI_Allreduce(local_gradient.data(), full_gradient.data(),
                  static_cast<int>(n_velocity), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    float total_misfit = 0.0f;
    MPI_Reduce(&local_misfit, &total_misfit, 1, MPI_FLOAT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      float grad_norm_sq = 0.0f;
      for (float g : full_gradient) grad_norm_sq += g * g;
      std::println("[starfwi-fwi] Iter {:3d}: misfit={:.6e}  |grad|={:.6e}",
                   iter + 1, total_misfit, std::sqrt(grad_norm_sq));
    }

    // -------- Velocity model update (gradient descent) --------
    // Normalize gradient by its max absolute value so that step_length has
    // physically meaningful units: max velocity change (m/s) per iteration.
    float grad_max = 0.0f;
    for (float g : full_gradient)
      grad_max = std::max(grad_max, std::abs(g));

    if (grad_max > 0.0f) {
      // All ranks apply the same update (full_gradient is identical everywhere
      // after MPI_Allreduce). We acquire the StarPU handle to notify it that
      // the underlying buffer is being modified.
      starpu_data_acquire(velocity_handle, STARPU_RW);
      const float scale = args.step_length / grad_max;
      for (size_t j = 0; j < n_velocity; ++j) {
        config.velocity_model.data[j] -= scale * full_gradient[j];
        // Keep velocities physically meaningful (>= 100 m/s)
        if (config.velocity_model.data[j] < 100.0f)
          config.velocity_model.data[j] = 100.0f;
      }
      starpu_data_release(velocity_handle);
      if (rank == 0)
        std::println("[starfwi-fwi]          max_grad={:.6e}  scale={:.6e}  "
                     "max_dv={:.2f} m/s",
                     grad_max, scale, args.step_length);
    }
  }

  // ========== STEP 8: SAVE FINAL INVERTED MODEL ==========
  if (rank == 0) {
    std::filesystem::create_directories(args.snapshot_dir);
    std::string out_model = args.snapshot_dir + "/inverted_velocity.bin";
    std::ofstream ofs(out_model, std::ios::binary);
    if (ofs) {
      ofs.write(reinterpret_cast<const char *>(config.velocity_model.data.data()),
                static_cast<std::streamsize>(n_velocity * sizeof(float)));
      std::println("[starfwi-fwi] Inverted model written to '{}'", out_model);
    } else {
      std::println(stderr, "[starfwi-fwi] WARNING: could not write inverted model to '{}'",
                   out_model);
    }
  }

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
    std::println("FWI inversion complete ({} iterations)!", args.num_iterations);
    std::println("Total time: {:.3f} s", max_us / 1e6);
    std::println("====================================");
  }

  starpu_mpi_checkpoint_shutdown();
  starpu_mpi_shutdown();
  return 0;
}
