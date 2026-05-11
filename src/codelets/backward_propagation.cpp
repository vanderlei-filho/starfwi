#include "codelets/backward_propagation.hpp"
#include "acoustics/finite_difference_solver.hpp"
#include "codelets/forward_propagation.hpp" // ShotData, TaskConfig, CodeletArg
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <print>
#include <semaphore.h>
#include <vector>

namespace starfwi {

// Backward propagation is CPU-only intentionally.
// REVOLVE checkpointing manages large pressure-field arrays in CPU RAM; routing
// backward tasks to CPU workers on all node types ensures identical memory
// behaviour and fair cross-instance comparison. The GPU (on g7e) is reserved
// exclusively for forward propagation where it provides the largest speedup.
struct starpu_codelet backward_propagation_codelet = {
    .cpu_funcs      = {backward_propagation_cpu},
    .cpu_funcs_name = {"backward_propagation_cpu"},
    .nbuffers = 6, // velocity, shot, config, receiver_x, receiver_y, receiver_z
    .modes = {STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
    // Keep shot (buf 1) and task_config (buf 2) in host RAM — they contain
    // std::vector members with heap data that StarPU cannot serialize to GPU.
    .specific_nodes = 1,
    .nodes = {STARPU_SPECIFIC_NODE_CPU,   // buf 0: velocity  → CPU (no GPU transfer)
              STARPU_SPECIFIC_NODE_CPU,   // buf 1: shot      → always host RAM
              STARPU_SPECIFIC_NODE_CPU,   // buf 2: config    → always host RAM
              STARPU_SPECIFIC_NODE_CPU,   // buf 3: recv_x    → CPU
              STARPU_SPECIFIC_NODE_CPU,   // buf 4: recv_y    → CPU
              STARPU_SPECIFIC_NODE_CPU},  // buf 5: recv_z    → CPU
    .name = "backward_propagation",
    .color = 0x00FF00};

// Counting semaphore that limits concurrent REVOLVE backward tasks to N workers.
// N is set by init_backward_semaphore() called from main_fwi.cpp before task
// submission. N is computed from the RAM budget: how many workers can each hold
// their seg_buf + solvers simultaneously without OOM.
// DISK/MEMORY backward paths do not acquire this semaphore — only REVOLVE does.
static sem_t           g_backward_sem;
static std::once_flag  g_backward_sem_init_flag;
static int             g_backward_max_parallel = 1; // default safe; overridden by init call

void init_backward_semaphore(int n) {
  g_backward_max_parallel = (n > 0) ? n : 1;
}

static void init_sem_once() {
  sem_init(&g_backward_sem, /*pshared=*/0,
           static_cast<unsigned int>(g_backward_max_parallel));
}

static inline size_t to_grid(float coord, float spacing, size_t max_idx) {
  if (spacing <= 0.0f) return 0;
  size_t idx = static_cast<size_t>(coord / spacing);
  return std::min(idx, max_idx - 1);
}

void backward_propagation_cpu(void *buffers[], void *cl_arg) {
  // ---- Extract StarPU buffers ----
  unsigned int n_velocity = STARPU_VECTOR_GET_NX(buffers[0]);
  const float *velocity_data = (const float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  ShotData *shot       = (ShotData *)STARPU_VARIABLE_GET_PTR(buffers[1]);
  const TaskConfig *task_config =
      (const TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[2]);
  const float *receiver_x = (const float *)STARPU_VECTOR_GET_PTR(buffers[3]);
  const float *receiver_y = (const float *)STARPU_VECTOR_GET_PTR(buffers[4]);
  const float *receiver_z = (const float *)STARPU_VECTOR_GET_PTR(buffers[5]);

  CodeletArg *arg = (CodeletArg *)cl_arg;
  const char *hostname = arg ? arg->hostname : "unknown";
  bool verbose = arg ? arg->verbose : false;

  // Log memory before semaphore wait — captures peak pressure from concurrent
  // forward workers still running alongside early backward tasks.
  {
    size_t rss_kb = 0, avail_kb = 0;
    if (std::ifstream f("/proc/self/status"); f) {
      std::string line;
      while (std::getline(f, line))
        if (line.starts_with("VmRSS:")) {
          std::sscanf(line.c_str(), "VmRSS: %zu kB", &rss_kb); break;
        }
    }
    if (std::ifstream f("/proc/meminfo"); f) {
      std::string line;
      while (std::getline(f, line))
        if (line.starts_with("MemAvailable:")) {
          std::sscanf(line.c_str(), "MemAvailable: %zu kB", &avail_kb); break;
        }
    }
    std::println("[MEM] host={} shot={} tag=bwd_entry rss_mb={} avail_mb={}",
                 hostname, shot->shot_id, rss_kb >> 10, avail_kb >> 10);
  }

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
  const bool use_memory  = (shot->wavefield_storage_actual == 0);
  const bool use_disk    = (shot->wavefield_storage_actual == 1);
  const bool use_hybrid  = (shot->wavefield_storage_actual == 2);
  const bool use_revolve = (shot->wavefield_storage_actual == 3);
  const size_t n_in_ram  = shot->snapshots_in_ram;
  const bool needs_file  = use_disk || use_hybrid;

  // Rolling buffer for disk/hybrid modes (p_hi, p_mid, p_lo = three adjacent snapshots).
  // Only allocated when actually needed (not in REVOLVE mode) — each is 152 MB+.
  std::ifstream wf_in;
  std::string wf_file;
  std::vector<float> p_hi, p_mid, p_lo;
  if (needs_file) {
    p_hi.assign(grid_size, 0.0f);
    p_mid.assign(grid_size, 0.0f);
    p_lo.assign(grid_size, 0.0f);
  }

  if (use_memory) {
    if (shot->pressure_snapshots.size() != nt * grid_size) {
      std::println(stderr,
                   "[starfwi][{}][backward_propagation_cpu] ERROR: Shot {} "
                   "pressure_snapshots size mismatch ({} vs expected {})",
                   hostname, shot->shot_id,
                   shot->pressure_snapshots.size(), nt * grid_size);
      return;
    }
  }

  if (needs_file) {
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
  }

  // Unified snapshot accessor for disk and hybrid modes.
  // DISK:   file contains t=0..nt-1, offset = t * grid_size * sizeof(float)
  // HYBRID: file contains t=n_in_ram..nt-1, offset = (t-n_in_ram) * grid_size * sizeof(float)
  //         RAM contains t=0..n_in_ram-1 in shot->pressure_snapshots
  long long disk_read_ms = 0; // declared here so read_snap can capture it
  auto read_snap = [&](size_t t, std::vector<float> &buf) {
    if (use_hybrid && t < n_in_ram) {
      const float *src = shot->pressure_snapshots.data() + t * grid_size;
      std::copy(src, src + grid_size, buf.data());
    } else {
      const size_t file_t = use_hybrid ? (t - n_in_ram) : t;
      auto t0 = std::chrono::high_resolution_clock::now();
      wf_in.seekg(static_cast<std::streamoff>(file_t * grid_size * sizeof(float)));
      wf_in.read(reinterpret_cast<char *>(buf.data()), grid_size * sizeof(float));
      disk_read_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }
  };

  if (needs_file) {
    // Prime the rolling buffer with the last two snapshots.
    // p_hi starts as zero (p[nt] ≈ 0, wave has left the domain).
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

  for (size_t s = 0; !use_revolve && s < nt; ++s) {
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
        // DISK or HYBRID: p_hi/p_mid/p_lo already loaded into rolling buffer.
        for (size_t i = 0; i < grid_size; ++i) {
          float p_ddot = (p_hi[i] - 2.0f * p_mid[i] + p_lo[i]) * dt2_inv;
          gradient[i] += (-2.0f * dt / c_cubed[i]) * p_ddot * q[i];
        }
      }
    }

    // Advance rolling buffer for disk/hybrid modes.
    if (needs_file && t_fwd >= 1) {
      std::swap(p_hi, p_mid);
      std::swap(p_mid, p_lo);
      if (t_fwd >= 2)
        read_snap(t_fwd - 2, p_lo);
      else
        std::fill(p_lo.begin(), p_lo.end(), 0.0f); // p[-1] ≈ 0
    }
  }

  // ================================================================
  // REVOLVE backward pass: segment-based recomputation, zero disk I/O.
  // For each segment (last → first): recompute the K forward steps from
  // the checkpoint, store them in a temp buffer, then walk backward
  // through the segment applying the imaging condition.
  // ================================================================
  if (use_revolve) {
    // Acquire one slot from the semaphore before any large allocations.
    // The slot is released when this segment of work is done (see sem_post below).
    // Parallel DISK/MEMORY backward tasks are unaffected — they never touch this.
    std::call_once(g_backward_sem_init_flag, init_sem_once);
    sem_wait(&g_backward_sem);

    const size_t K    = shot->revolve_segment_size;
    const size_t n_cp = shot->revolve_checkpoints.size();
    // Boundary sentinel — represents p[-1] ≈ 0 (wave not yet injected) and
    // p[nt] ≈ 0 (wave has left the domain). Allocated here (not at function
    // entry) since REVOLVE is the only path that needs it.
    const std::vector<float> zeros(grid_size, 0.0f);

    // Build SimulationConfig for the recomputation forward solver
    SimulationConfig fwd_config;
    fwd_config.grid.nx = nx; fwd_config.grid.ny = ny; fwd_config.grid.nz = nz;
    fwd_config.grid.dx = dx; fwd_config.grid.dy = dy; fwd_config.grid.dz = dz;
    fwd_config.time.dt = dt; fwd_config.time.nt = nt;
    fwd_config.velocity_model.data.assign(velocity_data, velocity_data + grid_size);

    // Process segments from last to first.
    // seg = n_cp-1: "tail" from last checkpoint to nt-1 (p[nt] ≈ 0 as cp_next)
    // seg = 0..n_cp-2: full segments between consecutive checkpoints
    for (int seg = (int)n_cp - 1; seg >= 0; --seg) {
      const size_t t_start = shot->revolve_checkpoint_times[seg];
      const size_t t_end   = (seg + 1 < (int)n_cp)
                               ? shot->revolve_checkpoint_times[seg + 1]
                               : nt;
      const size_t seg_len = t_end - t_start;
      if (seg_len == 0) continue;

      // Recompute forward from checkpoint[seg] to t_end (seg_len steps)
      FiniteDifferenceSolver fwd(fwd_config);
      fwd.initialize();
      fwd.set_source_position(shot->source_x, shot->source_y, shot->source_z);
      fwd.set_pressure_field(shot->revolve_checkpoints[seg].data(), grid_size);

      // seg_buf[i] = pressure field at t = t_start + i  (i = 0..seg_len)
      std::vector<std::vector<float>> seg_buf(seg_len + 1,
                                              std::vector<float>(grid_size));
      std::copy(shot->revolve_checkpoints[seg].begin(),
                shot->revolve_checkpoints[seg].end(),
                seg_buf[0].begin());
      for (size_t step = 1; step <= seg_len; ++step) {
        const size_t t = t_start + step;
        if (t > 0 && t - 1 < shot->source_wavelet.size())
          fwd.apply_source(shot->source_wavelet[t - 1], shot->shot_id);
        fwd.step();
        std::copy(fwd.get_pressure_field().begin(),
                  fwd.get_pressure_field().end(),
                  seg_buf[step].begin());
      }

      // p[t_end] ≈ 0 for the tail segment (wave has left the domain)
      const float *cp_next = (seg + 1 < (int)n_cp)
                               ? shot->revolve_checkpoints[seg + 1].data()
                               : zeros.data();

      // Walk backward through this segment, applying the imaging condition
      for (size_t step = seg_len; step-- > 0;) {
        const size_t t_fwd = t_start + step;

        // Advance adjoint solver
        for (size_t r = 0; r < n_receivers; ++r) {
          float adj_src = shot->residuals[r * nt + (nt - 1 - t_fwd)];
          if (adj_src != 0.0f)
            adj_solver.apply_source_at(adj_src, rx_ix[r], rx_iy[r], rx_iz[r]);
        }
        adj_solver.step();

        if (t_fwd >= 1 && t_fwd < nt - 1) {
          const float *ph = (step + 1 <= seg_len) ? seg_buf[step + 1].data()
                                                   : cp_next;
          const float *pm = seg_buf[step].data();
          const float *pl = (step > 0) ? seg_buf[step - 1].data() : zeros.data();
          const std::vector<float> &q = adj_solver.get_pressure_field();
          for (size_t i = 0; i < grid_size; ++i) {
            float p_ddot = (ph[i] - 2.0f * pm[i] + pl[i]) * dt2_inv;
            gradient[i] += (-2.0f * dt / c_cubed[i]) * p_ddot * q[i];
          }
        }
      }
    }

    // Release the semaphore slot — seg_buf and checkpoints have been freed
    // inside the segment loop, so peak REVOLVE memory has already dropped.
    sem_post(&g_backward_sem);
  }

  shot->gradient = std::move(gradient);

  // Free snapshot memory and disk file now that they're no longer needed.
  shot->pressure_snapshots.clear();
  shot->pressure_snapshots.shrink_to_fit();
  shot->revolve_checkpoints.clear();
  shot->revolve_checkpoints.shrink_to_fit();
  shot->revolve_checkpoint_times.clear();
  if (needs_file) {
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

  const std::string storage_str = use_memory   ? "ram"
                                 : use_revolve ? "revolve"
                                 : use_hybrid  ? "hybrid"
                                              : "disk";
  std::println("[METRIC] shot={} bwd_total_ms={} bwd_compute_ms={} "
               "bwd_disk_read_ms={} storage={}",
               shot->shot_id, total_ms, total_ms - disk_read_ms,
               disk_read_ms, storage_str);
}

} // namespace starfwi
