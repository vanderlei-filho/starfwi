#include "codelets/backward_propagation.hpp"
#include "acoustics/finite_difference_solver.hpp"
#include "codelets/forward_propagation.hpp" // ShotData, TaskConfig, CodeletArg
#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <vector>

namespace starfwi {

struct starpu_codelet backward_propagation_codelet = {
    .cpu_funcs = {backward_propagation_cpu},
    .cpu_funcs_name = {"backward_propagation_cpu"},
    .nbuffers = 6, // velocity, shot, config, receiver_x, receiver_y, receiver_z
    .modes = {STARPU_R, STARPU_RW, STARPU_R, STARPU_R, STARPU_R, STARPU_R},
    .name = "backward_propagation",
    .color = 0x00FF00};

static inline size_t to_grid(float coord, float spacing, size_t max_idx) {
  if (spacing <= 0.0f) return 0;
  size_t idx = static_cast<size_t>(coord / spacing);
  return std::min(idx, max_idx - 1);
}

// Compute p[t-1] from p[t] and p[t+1] via the time-reversed acoustic wave
// equation (valid for the dissipation-free acoustic equation):
//   p[t-1] = 2·p[t] - p[t+1] + (c·dt)² · ∇²p[t]
// Uses 2nd-order centred FD for the Laplacian. Boundaries are Dirichlet (zero).
static void backward_wave_step_2d(std::vector<float> &p_out,
                                   const std::vector<float> &p_curr,
                                   const std::vector<float> &p_next,
                                   const std::vector<float> &vel_sq,
                                   float dt2, size_t nx, size_t nz,
                                   float dx2_inv, float dz2_inv) {
  std::fill(p_out.begin(), p_out.end(), 0.0f);
  for (size_t iz = 1; iz < nz - 1; ++iz) {
    size_t base = nx * iz;
    for (size_t ix = 1; ix < nx - 1; ++ix) {
      size_t i = base + ix;
      float lap = (p_curr[i - 1] - 2.0f * p_curr[i] + p_curr[i + 1]) * dx2_inv
                + (p_curr[i - nx] - 2.0f * p_curr[i] + p_curr[i + nx]) * dz2_inv;
      p_out[i] = 2.0f * p_curr[i] - p_next[i] + vel_sq[i] * dt2 * lap;
    }
  }
}

static void backward_wave_step_3d(std::vector<float> &p_out,
                                   const std::vector<float> &p_curr,
                                   const std::vector<float> &p_next,
                                   const std::vector<float> &vel_sq,
                                   float dt2, size_t nx, size_t ny, size_t nz,
                                   float dx2_inv, float dy2_inv, float dz2_inv) {
  std::fill(p_out.begin(), p_out.end(), 0.0f);
  for (size_t iz = 1; iz < nz - 1; ++iz) {
    for (size_t iy = 1; iy < ny - 1; ++iy) {
      size_t base = nx * (iy + ny * iz);
      for (size_t ix = 1; ix < nx - 1; ++ix) {
        size_t i = base + ix;
        float lap =
            (p_curr[i - 1]     - 2.0f * p_curr[i] + p_curr[i + 1])     * dx2_inv
          + (p_curr[i - nx]    - 2.0f * p_curr[i] + p_curr[i + nx])    * dy2_inv
          + (p_curr[i - nx*ny] - 2.0f * p_curr[i] + p_curr[i + nx*ny]) * dz2_inv;
        p_out[i] = 2.0f * p_curr[i] - p_next[i] + vel_sq[i] * dt2 * lap;
      }
    }
  }
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
    if (verbose) {
      std::println("[starfwi][{}][backward_propagation_cpu] Shot {} has no "
                   "residuals, skipping adjoint propagation",
                   hostname, shot->shot_id);
    }
    return;
  }

  const size_t nt = task_config->nt;
  const size_t n_receivers = task_config->n_receivers;
  const size_t grid_size = n_velocity;
  const float dt = task_config->dt;
  const float dt2 = dt * dt;
  const float dt2_inv = 1.0f / dt2;

  const size_t nx = task_config->nx;
  const size_t ny = task_config->ny;
  const size_t nz = task_config->nz;
  const float dx = task_config->dx;
  const float dy = task_config->dy;
  const float dz = task_config->dz;

  const bool is_2d = (ny == 1);
  const float dx2_inv = 1.0f / (dx * dx);
  const float dz2_inv = 1.0f / (dz * dz);
  const float dy2_inv = (!is_2d && dy > 0.0f) ? 1.0f / (dy * dy) : 0.0f;

  auto start_time = std::chrono::high_resolution_clock::now();

  if (verbose) {
    std::println("[starfwi][{}][backward_propagation_cpu] Shot {}: starting "
                 "adjoint propagation (nt={}, grid={})",
                 hostname, shot->shot_id, nt, grid_size);
  }

  // ---- Build SimulationConfig for the FD solver ----
  SimulationConfig config;
  config.grid.nx = nx;
  config.grid.ny = ny;
  config.grid.nz = nz;
  config.grid.dx = dx;
  config.grid.dy = dy;
  config.grid.dz = dz;
  config.time.dt = dt;
  config.time.nt = nt;
  config.velocity_model.data.assign(velocity_data, velocity_data + grid_size);

  // Pre-compute vel² (for backward reconstruction) and c³ (for gradient).
  std::vector<float> vel_sq(grid_size);
  std::vector<float> c_cubed(grid_size);
  for (size_t i = 0; i < grid_size; ++i) {
    float c = velocity_data[i];
    vel_sq[i] = c * c;
    c_cubed[i] = c * vel_sq[i];
  }

  // ---- Pre-compute receiver grid indices ----
  std::vector<size_t> rx_ix(n_receivers), rx_iy(n_receivers), rx_iz(n_receivers);
  for (size_t r = 0; r < n_receivers; ++r) {
    rx_ix[r] = to_grid(receiver_x[r], dx, nx);
    rx_iy[r] = is_2d ? 0 : to_grid(receiver_y[r], dy, ny);
    rx_iz[r] = to_grid(receiver_z[r], dz, nz);
  }

  // ================================================================
  // PASS 1: Re-run forward simulation, keeping only the last 3
  // pressure snapshots in a rolling buffer.
  //
  // Memory: 3 × grid_size floats ≈ 430 MB for Marmousi2 (38M cells),
  // versus nt × grid_size ≈ 14.4 GB with the naive all-snapshot approach.
  // ================================================================
  std::vector<std::vector<float>> p_roll(3, std::vector<float>(grid_size, 0.0f));

  {
    FiniteDifferenceSolver fwd_solver(config);
    if (!fwd_solver.initialize()) {
      std::println(stderr,
                   "[starfwi][{}][backward_propagation_cpu] ERROR: forward "
                   "re-run init failed for shot {}",
                   hostname, shot->shot_id);
      return;
    }
    fwd_solver.set_source_position(shot->source_x, shot->source_y,
                                   shot->source_z);

    for (size_t t = 0; t < nt; ++t) {
      if (t < shot->source_wavelet.size()) {
        fwd_solver.apply_source(shot->source_wavelet[t], shot->shot_id);
      }
      fwd_solver.step();
      p_roll[t % 3] = fwd_solver.get_pressure_field(); // p[t]
    }
  }

  if (verbose) {
    std::println("[starfwi][{}][backward_propagation_cpu] Shot {}: forward "
                 "re-run complete",
                 hostname, shot->shot_id);
  }

  // ================================================================
  // PASS 2: Adjoint propagation with time-reversed forward field.
  //
  // The acoustic wave equation is time-symmetric (no dissipation), so
  // p[t−1] can be reconstructed from p[t] and p[t+1] using the same FD
  // stencil:
  //   p[t−1] = 2·p[t] − p[t+1] + (c·dt)²·∇²p[t]
  //
  // Three rolling states are maintained:
  //   p_hi  — p[t_fwd + 1]  (one step ahead in physical time)
  //   p_mid — p[t_fwd]      (current physical time, used for cross-corr)
  //   p_lo  — p[t_fwd − 1]  (reconstructed via backward wave equation)
  //
  // At adjoint step s, t_fwd = nt−1−s.  The gradient accumulates:
  //   grad[i] += (−2/c³[i]) · p̈[i] · q[i] · dt
  //   p̈ = (p_hi[i] − 2·p_mid[i] + p_lo[i]) / dt²
  // ================================================================
  std::vector<float> gradient(grid_size, 0.0f);

  // Initialise backward window from the last 3 forward snapshots.
  // p[nt] is not computed — approximate as zero (boundary assumption).
  std::vector<float> p_hi(grid_size, 0.0f);              // p[nt] ≈ 0
  std::vector<float> p_mid = p_roll[(nt - 1) % 3];       // p[nt−1]
  std::vector<float> p_lo  = p_roll[(nt - 2 + 3) % 3];  // p[nt−2]
  std::vector<float> p_lo_next(grid_size, 0.0f);         // workspace

  {
    FiniteDifferenceSolver adj_solver(config);
    if (!adj_solver.initialize()) {
      std::println(stderr,
                   "[starfwi][{}][backward_propagation_cpu] ERROR: adjoint "
                   "solver init failed for shot {}",
                   hostname, shot->shot_id);
      return;
    }

    for (size_t s = 0; s < nt; ++s) {
      // Physical forward time corresponding to this adjoint step.
      size_t t_fwd = nt - 1 - s;

      // Inject adjoint sources at all receiver positions.
      // residuals layout: residuals[receiver * nt + adjoint_time_step]
      for (size_t r = 0; r < n_receivers; ++r) {
        float adj_src = shot->residuals[r * nt + s];
        if (adj_src != 0.0f) {
          adj_solver.apply_source_at(adj_src, rx_ix[r], rx_iy[r], rx_iz[r]);
        }
      }

      adj_solver.step();

      // Accumulate gradient (skip the two boundary timesteps where p̈ is
      // not well-defined: t_fwd=0 needs p[−1], t_fwd=nt−1 needs p[nt]).
      if (t_fwd >= 1 && t_fwd < nt - 1) {
        const std::vector<float> &q = adj_solver.get_pressure_field();
        for (size_t i = 0; i < grid_size; ++i) {
          float p_ddot =
              (p_hi[i] - 2.0f * p_mid[i] + p_lo[i]) * dt2_inv;
          gradient[i] += (-2.0f * dt / c_cubed[i]) * p_ddot * q[i];
        }
      }

      // Advance rolling window for the next outer step (t_fwd → t_fwd−1).
      if (t_fwd >= 1) {
        std::vector<float> p_hi_new  = std::move(p_mid);
        std::vector<float> p_mid_new = std::move(p_lo);

        if (s == 0) {
          // p[nt−3] is the third state in the rolling buffer — exact.
          p_lo = p_roll[(nt - 3 + 3) % 3];
        } else if (t_fwd >= 2) {
          // Reconstruct p[t_fwd−2] via backward wave equation applied to
          // p_mid_new (=p[t_fwd−1]) and p_hi_new (=p[t_fwd]).
          if (is_2d) {
            backward_wave_step_2d(p_lo_next, p_mid_new, p_hi_new, vel_sq,
                                  dt2, nx, nz, dx2_inv, dz2_inv);
          } else {
            backward_wave_step_3d(p_lo_next, p_mid_new, p_hi_new, vel_sq,
                                  dt2, nx, ny, nz, dx2_inv, dy2_inv, dz2_inv);
          }
          p_lo = p_lo_next;
        }

        p_hi  = std::move(p_hi_new);
        p_mid = std::move(p_mid_new);
      }
    }
  }

  shot->gradient = std::move(gradient);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  if (verbose) {
    std::println("[starfwi][{}][backward_propagation_cpu] Shot {} adjoint "
                 "propagation completed in {:.3f} s",
                 hostname, shot->shot_id, duration.count() / 1000.0);
  }
}

} // namespace starfwi
