#include "codelets/backward_propagation.hpp"
#include "acoustics/finite_difference_solver.hpp"
#include "codelets/forward_propagation.hpp" // ShotData, TaskConfig, CodeletArg
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <vector>

namespace starfwi {

struct starpu_codelet backward_propagation_codelet = {
    .cpu_funcs      = {backward_propagation_cpu},
#ifdef STARPU_USE_CUDA
    .cuda_funcs     = {backward_propagation_cuda},
    .cuda_flags     = {STARPU_CUDA_ASYNC},
#endif
    .cpu_funcs_name = {"backward_propagation_cpu"},
    .nbuffers = 6, // velocity, shot, config, receiver_x, receiver_y, receiver_z
    .modes = {STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
    // Same reasoning as forward_propagation_codelet: keep shot (buf 1) and
    // task_config (buf 2) in host RAM. See forward_propagation.cpp for details.
    .specific_nodes = 1,
    .nodes = {STARPU_SPECIFIC_NODE_LOCAL,  // buf 0: velocity → GPU for CUDA
              STARPU_SPECIFIC_NODE_CPU,    // buf 1: shot     → always host RAM
              STARPU_SPECIFIC_NODE_CPU,    // buf 2: config   → always host RAM
              STARPU_SPECIFIC_NODE_LOCAL,  // buf 3: recv_x   → GPU for CUDA
              STARPU_SPECIFIC_NODE_LOCAL,  // buf 4: recv_y   → GPU for CUDA
              STARPU_SPECIFIC_NODE_LOCAL}, // buf 5: recv_z   → GPU for CUDA
    .name = "backward_propagation",
    .color = 0x00FF00};

static inline size_t to_grid(float coord, float spacing, size_t max_idx) {
  if (spacing <= 0.0f) return 0;
  size_t idx = static_cast<size_t>(coord / spacing);
  return std::min(idx, max_idx - 1);
}

void backward_propagation_cpu(void *buffers[], void *cl_arg) {
  // ---- Extract StarPU buffers ----
  unsigned int n_velocity = STARPU_VECTOR_GET_NX(buffers[0]);
  const float *velocity_data = (const float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  ShotData *shot = (ShotData *)STARPU_VARIABLE_GET_PTR(buffers[1]);
  const TaskConfig *task_config =
      (const TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[2]);
  const float *receiver_x = (const float *)STARPU_VECTOR_GET_PTR(buffers[3]);
  const float *receiver_y = (const float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  const float *receiver_z = (const float *)STARPU_VECTOR_GET_PTR(buffers[5]);

  CodeletArg *arg = (CodeletArg *)cl_arg;
  const char *hostname = arg ? arg->hostname : "unknown";
  bool verbose = arg ? arg->verbose : false;

  // Skip if there are no residuals (misfit was not computed for this shot).
  if (shot->residuals.empty()) {
    if (verbose)
      std::println("[starfwi][{}][backward_propagation_cpu] Shot {} has no "
                   "residuals, skipping adjoint propagation",
                   hostname, shot->shot_id);
    return;
  }

  const size_t nt         = task_config->nt;
  const size_t n_receivers = task_config->n_receivers;
  const size_t grid_size  = n_velocity;
  const float  dt         = task_config->dt;
  const float  dt2_inv    = 1.0f / (dt * dt);

  const size_t nx = task_config->nx;
  const size_t ny = task_config->ny;
  const size_t nz = task_config->nz;
  const float  dx = task_config->dx;
  const float  dy = task_config->dy;
  const float  dz = task_config->dz;
  const bool is_2d = (ny == 1);

  auto start_time = std::chrono::high_resolution_clock::now();

  if (verbose)
    std::println("[starfwi][{}][backward_propagation_cpu] Shot {}: starting "
                 "adjoint propagation (nt={}, grid={})",
                 hostname, shot->shot_id, nt, grid_size);

  // ---- Pre-compute c³ for the imaging condition ----
  std::vector<float> c_cubed(grid_size);
  for (size_t i = 0; i < grid_size; ++i) {
    float c = velocity_data[i];
    c_cubed[i] = c * c * c;
  }

  // ---- Pre-compute receiver grid indices ----
  std::vector<size_t> rx_ix(n_receivers), rx_iy(n_receivers), rx_iz(n_receivers);
  for (size_t r = 0; r < n_receivers; ++r) {
    rx_ix[r] = to_grid(receiver_x[r], dx, nx);
    rx_iy[r] = is_2d ? 0 : to_grid(receiver_y[r], dy, ny);
    rx_iz[r] = to_grid(receiver_z[r], dz, nz);
  }

  // ================================================================
  // Snapshot access helpers.
  //
  // MEMORY strategy: pressure_snapshots holds all nt snapshots inline.
  //   snap(t) returns a direct pointer into that buffer — zero copies.
  //
  // DISK strategy: snapshots were written to a binary file by the forward
  //   codelet. We use a 3-slot rolling buffer (p_hi, p_mid, p_lo) and
  //   read one new snapshot from disk per adjoint step.
  // ================================================================
  // Use the storage mode actually chosen by the forward codelet at runtime.
  // This may differ from task_config->wavefield_storage when auto-detection
  // overrode the requested mode based on available GPU/host memory.
  const bool use_memory = (shot->wavefield_storage_actual == 0);

  // DISK: open wavefield file and set up rolling buffer
  std::ifstream wf_in;
  std::string wf_file;
  std::vector<float> p_hi (grid_size, 0.0f); // p[t_fwd+1], init = p[nt] ≈ 0
  std::vector<float> p_mid(grid_size, 0.0f); // p[t_fwd]
  std::vector<float> p_lo (grid_size, 0.0f); // p[t_fwd-1]

  if (use_memory) {
    if (shot->pressure_snapshots.size() != nt * grid_size) {
      std::println(stderr,
                   "[starfwi][{}][backward_propagation_cpu] ERROR: Shot {} "
                   "pressure_snapshots size mismatch ({} vs expected {})",
                   hostname, shot->shot_id,
                   shot->pressure_snapshots.size(), nt * grid_size);
      return;
    }
  } else {
    // DISK: open the file written by the forward codelet
    wf_file = std::format("{}/fwd_shot_{}.bin",
                           task_config->wavefield_dir, shot->shot_id);
    wf_in.open(wf_file, std::ios::binary);
    if (!wf_in) {
      std::println(stderr,
                   "[starfwi][{}][backward_propagation_cpu] ERROR: Shot {} "
                   "cannot open wavefield file '{}'",
                   hostname, shot->shot_id, wf_file);
      return;
    }

    // Prime the rolling buffer with the last two snapshots.
    // p_hi starts as zero (p[nt] ≈ 0, wave has left the domain).
    auto read_snap = [&](size_t t, std::vector<float> &buf) {
      wf_in.seekg(static_cast<std::streamoff>(t * grid_size * sizeof(float)));
      wf_in.read(reinterpret_cast<char *>(buf.data()), grid_size * sizeof(float));
    };
    read_snap(nt - 1, p_mid);
    if (nt >= 2) read_snap(nt - 2, p_lo);
  }

  // ================================================================
  // Adjoint propagation — single pass, backward in time.
  //
  // At each step s, t_fwd = nt-1-s is the corresponding forward time.
  // We inject the time-reversed residuals as adjoint sources and
  // cross-correlate with the forward field to accumulate the gradient:
  //
  //   grad[i] += (−2·dt / c³[i]) · p̈[i,t_fwd] · q[i,t_fwd]
  //   p̈ = (p[t_fwd+1] − 2·p[t_fwd] + p[t_fwd−1]) / dt²
  // ================================================================
  std::vector<float> gradient(grid_size, 0.0f);
  long long disk_read_ms = 0;

  SimulationConfig config;
  config.grid.nx = nx; config.grid.ny = ny; config.grid.nz = nz;
  config.grid.dx = dx; config.grid.dy = dy; config.grid.dz = dz;
  config.time.dt = dt; config.time.nt = nt;
  config.velocity_model.data.assign(velocity_data, velocity_data + grid_size);

  FiniteDifferenceSolver adj_solver(config);
  if (!adj_solver.initialize()) {
    std::println(stderr,
                 "[starfwi][{}][backward_propagation_cpu] ERROR: adjoint "
                 "solver init failed for shot {}",
                 hostname, shot->shot_id);
    return;
  }

  for (size_t s = 0; s < nt; ++s) {
    const size_t t_fwd = nt - 1 - s;

    // Inject adjoint sources at receiver positions (time-reversed residuals).
    // residuals layout: residuals[receiver * nt + adjoint_time_step]
    for (size_t r = 0; r < n_receivers; ++r) {
      float adj_src = shot->residuals[r * nt + s];
      if (adj_src != 0.0f)
        adj_solver.apply_source_at(adj_src, rx_ix[r], rx_iy[r], rx_iz[r]);
    }

    adj_solver.step();

    // Accumulate gradient (skip boundary timesteps where p̈ is not well-defined).
    if (t_fwd >= 1 && t_fwd < nt - 1) {
      const std::vector<float> &q = adj_solver.get_pressure_field();

      if (use_memory) {
        // Zero-copy: point directly into the snapshot buffer.
        const float *ph = shot->pressure_snapshots.data() + (t_fwd + 1) * grid_size;
        const float *pm = shot->pressure_snapshots.data() +  t_fwd      * grid_size;
        const float *pl = shot->pressure_snapshots.data() + (t_fwd - 1) * grid_size;
        for (size_t i = 0; i < grid_size; ++i) {
          float p_ddot = (ph[i] - 2.0f * pm[i] + pl[i]) * dt2_inv;
          gradient[i] += (-2.0f * dt / c_cubed[i]) * p_ddot * q[i];
        }
      } else {
        // DISK: p_hi/p_mid/p_lo already loaded into rolling buffer.
        for (size_t i = 0; i < grid_size; ++i) {
          float p_ddot = (p_hi[i] - 2.0f * p_mid[i] + p_lo[i]) * dt2_inv;
          gradient[i] += (-2.0f * dt / c_cubed[i]) * p_ddot * q[i];
        }
      }
    }

    // Advance DISK rolling buffer: shift window and read next snapshot.
    if (!use_memory && t_fwd >= 1) {
      std::swap(p_hi, p_mid);
      std::swap(p_mid, p_lo);
      auto t0 = std::chrono::high_resolution_clock::now();
      if (t_fwd >= 2) {
        wf_in.seekg(static_cast<std::streamoff>((t_fwd - 2) * grid_size * sizeof(float)));
        wf_in.read(reinterpret_cast<char *>(p_lo.data()), grid_size * sizeof(float));
      } else {
        std::fill(p_lo.begin(), p_lo.end(), 0.0f); // p[-1] ≈ 0
      }
      disk_read_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }
  }

  shot->gradient = std::move(gradient);

  // Free snapshot memory now that it's no longer needed.
  if (use_memory) {
    shot->pressure_snapshots.clear();
    shot->pressure_snapshots.shrink_to_fit();
  } else {
    wf_in.close();
    std::filesystem::remove(wf_file);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  if (verbose)
    std::println("[starfwi][{}][backward_propagation_cpu] Shot {} adjoint "
                 "propagation completed in {:.3f} s",
                 hostname, shot->shot_id, total_ms / 1000.0);

  std::println("[METRIC] shot={} bwd_total_ms={} bwd_compute_ms={} "
               "bwd_disk_read_ms={} storage={}",
               shot->shot_id, total_ms, total_ms - disk_read_ms,
               disk_read_ms, use_memory ? "ram" : "disk");
}

} // namespace starfwi
