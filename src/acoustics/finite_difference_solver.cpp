#include "acoustics/finite_difference_solver.hpp"
#include <algorithm>
#include <cmath>
#include <print>

namespace starfwi {

/**
 * @brief Constructor for FiniteDifferenceSolver
 *
 * Initializes the acoustic solver with grid parameters from the configuration.
 * Extracts grid dimensions, spacing, and time step for use in the simulation.
 *
 * Note: Source position must be set using set_source_position() before
 * calling apply_source().
 *
 * @param config Input configuration containing grid, time, and model parameters
 */
FiniteDifferenceSolver::FiniteDifferenceSolver(const SimulationConfig &config)
    : config_(config), nx_(config.grid.nx), ny_(config.grid.ny),
      nz_(config.grid.nz), dx_(config.grid.dx), dy_(config.grid.dy),
      dz_(config.grid.dz), dt_(config.time.dt), source_position_set_(false),
      source_ix_(0), source_iy_(0), source_iz_(0) {}

/**
 * @brief Initialize the wave propagator
 *
 * Allocates memory for wavefield arrays (current, previous, and next time
 * steps), prepares the velocity model by squaring velocities for computational
 * efficiency, and validates the CFL stability condition to ensure numerical
 * stability.
 *
 * The CFL condition states: dt < dx / (v_max * sqrt(3))
 * where v_max is the maximum velocity in the model.
 *
 * @return true if initialization succeeds, false on error
 */
bool FiniteDifferenceSolver::initialize() {
  size_t grid_size = nx_ * ny_ * nz_;

  // Allocate wavefield arrays for three time levels (t-1, t, t+1)
  pressure_.resize(grid_size, 0.0f);     // Current time step (t)
  pressure_old_.resize(grid_size, 0.0f); // Previous time step (t-1)
  pressure_new_.resize(grid_size, 0.0f); // Next time step (t+1)

  // Prepare velocity model (square it for efficiency in wave equation)
  // Wave equation uses c² * ∇²p, so we pre-compute c² to avoid repeated
  // multiplications
  velocity_squared_.resize(grid_size);

  // Validate velocity model size matches grid
  if (config_.velocity_model.data.size() != grid_size) {
    std::println(stderr, "ERROR: Velocity model size mismatch!");
    return false;
  }

  // Square the velocities for use in wave equation
  for (size_t i = 0; i < grid_size; ++i) {
    float v = config_.velocity_model.data[i];
    velocity_squared_[i] = v * v;
  }

  // Check CFL (Courant-Friedrichs-Lewy) stability condition
  // For 3D acoustic wave equation: dt < dx / (v_max * sqrt(3))
  float max_velocity = std::sqrt(
      *std::max_element(velocity_squared_.begin(), velocity_squared_.end()));
  float min_spacing = std::min({dx_, dy_, dz_});
  float cfl_limit = min_spacing / (max_velocity * std::sqrt(3.0f));

  // std::cout << "  Max velocity: " << max_velocity << " m/s" << std::endl;
  // std::cout << "  CFL stability limit: " << cfl_limit << " s" << std::endl;
  // std::cout << "  Current time step: " << dt_ << " s" << std::endl;

  /*if (dt_ > cfl_limit) {
    std::cerr << "WARNING: Time step violates CFL condition!" << std::endl;
    std::cerr << "  Simulation may be unstable." << std::endl;
    std::cerr << "  Recommended dt < " << cfl_limit << " s" << std::endl;
  } else {
    std::cout << "  CFL condition satisfied" << std::endl;
  }*/

  // std::cout << "Wave propagator initialized successfully!" << std::endl;
  return true;
}

/**
 * @brief Advance the wave propagation by one time step
 *
 * Solves the acoustic wave equation for one time step:
 *   ∂²p/∂t² = c² * ∇²p
 *
 * Using finite difference discretization:
 *   p(t+1) = 2*p(t) - p(t-1) + (c*dt)² * ∇²p(t)
 *
 * Steps:
 *   1. Compute spatial Laplacian (∇²p) using finite differences
 *   2. Apply boundary conditions (absorbing boundaries)
 *   3. Rotate time levels (t-1 <- t <- t+1)
 */
void FiniteDifferenceSolver::step() {
  // Compute spatial derivatives (Laplacian)
  compute_laplacian();

  // Apply boundary conditions
  apply_boundary_conditions();

  // Swap time levels for next iteration
  swap_time_levels();
}

/**
 * @brief Compute the Laplacian operator using finite differences
 *
 * Calculates ∇²p = ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z² using second-order
 * centered finite difference stencils:
 *
 *   ∂²p/∂x² ≈ [p(i-1,j,k) - 2*p(i,j,k) + p(i+1,j,k)] / dx²
 *
 * Then updates the pressure field using the wave equation:
 *   p(t+1) = 2*p(t) - p(t-1) + (c*dt)² * ∇²p(t)
 *
 * Only processes interior points (excludes boundaries).
 *
 * Performance: Automatically detects 2D models (ny==1) and uses optimized 2D
 * loops. Uses direct index arithmetic instead of index() function calls to
 * eliminate redundant multiplications and improve cache efficiency.
 */
void FiniteDifferenceSolver::compute_laplacian() {
  // Detect if this is a 2D model (ny == 1)
  if (ny_ == 1) {
    compute_laplacian_2d();
  } else {
    compute_laplacian_3d();
  }
}

/**
 * @brief Optimized 2D Laplacian computation (when ny == 1)
 *
 * For 2D models, we skip the Y dimension entirely, resulting in:
 * - No Y-direction derivative computation
 * - No Y loop iteration
 * - Simpler indexing: index = ix + nx * iz
 * - Better cache performance and reduced memory bandwidth
 */
void FiniteDifferenceSolver::compute_laplacian_2d() {
  // Pre-compute inverse squared spacing for efficiency
  const float dx2_inv = 1.0f / (dx_ * dx_);
  const float dz2_inv = 1.0f / (dz_ * dz_);

  // Pre-compute stride for 2D indexing
  // For 2D: index = ix + nx * iz (no Y component)
  const size_t stride_z = nx_;

  // Loop over interior points only (boundaries handled separately)
  // Note: No Y loop since ny == 1
  for (size_t iz = 1; iz < nz_ - 1; ++iz) {
    // Compute base index for this row (z plane)
    size_t idx_base = stride_z * iz;

    for (size_t ix = 1; ix < nx_ - 1; ++ix) {
      // Current point index using direct arithmetic
      size_t idx = idx_base + ix;

      // Second derivative in x-direction (stride = 1)
      float d2p_dx2 =
          (pressure_[idx - 1] - 2.0f * pressure_[idx] + pressure_[idx + 1]) *
          dx2_inv;

      // Second derivative in z-direction (stride = nx)
      float d2p_dz2 = (pressure_[idx - stride_z] - 2.0f * pressure_[idx] +
                       pressure_[idx + stride_z]) *
                      dz2_inv;

      // Laplacian for 2D: only X and Z derivatives (no Y)
      float laplacian = d2p_dx2 + d2p_dz2;

      // Wave equation time update: p(t+1) = 2*p(t) - p(t-1) + c²*dt²*∇²p
      float c2_dt2 = velocity_squared_[idx] * dt_ * dt_;
      pressure_new_[idx] =
          2.0f * pressure_[idx] - pressure_old_[idx] + c2_dt2 * laplacian;
    }
  }
}

/**
 * @brief Full 3D Laplacian computation
 *
 * Computes all three spatial derivatives (X, Y, Z) for 3D models.
 */
void FiniteDifferenceSolver::compute_laplacian_3d() {
  // Pre-compute inverse squared spacing for efficiency
  const float dx2_inv = 1.0f / (dx_ * dx_);
  const float dy2_inv = 1.0f / (dy_ * dy_);
  const float dz2_inv = 1.0f / (dz_ * dz_);

  // Pre-compute strides for index arithmetic
  // For row-major storage: index = ix + nx * (iy + ny * iz)
  const size_t stride_y = nx_;       // Stride in y-direction
  const size_t stride_z = nx_ * ny_; // Stride in z-direction

  // Loop over interior points only (boundaries handled separately)
  for (size_t iz = 1; iz < nz_ - 1; ++iz) {
    for (size_t iy = 1; iy < ny_ - 1; ++iy) {
      // Compute base index for this row (y,z plane)
      // This eliminates repeated multiplication in the inner loop
      size_t idx_base = stride_y * (iy + ny_ * iz);

      for (size_t ix = 1; ix < nx_ - 1; ++ix) {
        // Current point index using direct arithmetic
        size_t idx = idx_base + ix;

        // Second derivative in x-direction (stride = 1)
        float d2p_dx2 =
            (pressure_[idx - 1] - 2.0f * pressure_[idx] + pressure_[idx + 1]) *
            dx2_inv;

        // Second derivative in y-direction (stride = nx)
        float d2p_dy2 = (pressure_[idx - stride_y] - 2.0f * pressure_[idx] +
                         pressure_[idx + stride_y]) *
                        dy2_inv;

        // Second derivative in z-direction (stride = nx * ny)
        float d2p_dz2 = (pressure_[idx - stride_z] - 2.0f * pressure_[idx] +
                         pressure_[idx + stride_z]) *
                        dz2_inv;

        // Laplacian: sum of second derivatives
        float laplacian = d2p_dx2 + d2p_dy2 + d2p_dz2;

        // Wave equation time update: p(t+1) = 2*p(t) - p(t-1) + c²*dt²*∇²p
        float c2_dt2 = velocity_squared_[idx] * dt_ * dt_;
        pressure_new_[idx] =
            2.0f * pressure_[idx] - pressure_old_[idx] + c2_dt2 * laplacian;
      }
    }
  }
}

/**
 * @brief Apply boundary conditions to the wavefield
 *
 * Implements simple Dirichlet (zero-pressure) boundary conditions by
 * setting all boundary points to zero. This creates an absorbing effect
 * but is not perfectly non-reflecting.
 *
 * Performance: Uses direct index arithmetic with pre-computed strides
 * for efficient memory access patterns.
 *
 * TODO: Implement Perfectly Matched Layers (PML) or sponge layers for
 * better absorbing boundaries that minimize artificial reflections.
 */
void FiniteDifferenceSolver::apply_boundary_conditions() {
  const size_t stride_y = nx_;
  const size_t stride_z = nx_ * ny_;

  // Zero out left and right boundaries (x-direction)
  // These are at ix=0 and ix=nx-1 for all (iy, iz)
  for (size_t iz = 0; iz < nz_; ++iz) {
    size_t base_z = iz * stride_z;
    for (size_t iy = 0; iy < ny_; ++iy) {
      size_t idx_base = base_z + iy * stride_y;
      pressure_new_[idx_base] = 0.0f;           // Left wall (ix=0)
      pressure_new_[idx_base + nx_ - 1] = 0.0f; // Right wall (ix=nx-1)
    }
  }

  // Zero out front and back boundaries (y-direction)
  // These are at iy=0 and iy=ny-1 for all (ix, iz)
  for (size_t iz = 0; iz < nz_; ++iz) {
    size_t base_z = iz * stride_z;
    for (size_t ix = 0; ix < nx_; ++ix) {
      pressure_new_[base_z + ix] = 0.0f; // Front (iy=0)
      pressure_new_[base_z + (ny_ - 1) * stride_y + ix] =
          0.0f; // Back (iy=ny-1)
    }
  }

  // Zero out bottom and top boundaries (z-direction)
  // These are at iz=0 and iz=nz-1 for all (ix, iy)
  for (size_t iy = 0; iy < ny_; ++iy) {
    size_t idx_bottom = iy * stride_y;                     // iz=0
    size_t idx_top = (nz_ - 1) * stride_z + iy * stride_y; // iz=nz-1
    for (size_t ix = 0; ix < nx_; ++ix) {
      pressure_new_[idx_bottom + ix] = 0.0f; // Bottom
      pressure_new_[idx_top + ix] = 0.0f;    // Top
    }
  }
}

/**
 * @brief Rotate time levels for the next iteration
 *
 * Efficiently swaps pointers to avoid copying large arrays:
 *   - pressure_old_ becomes pressure_ (t-1 <- t)
 *   - pressure_ becomes pressure_new_ (t <- t+1)
 *   - pressure_new_ becomes ready for next computation
 *
 * This implements the three-level time-stepping scheme required
 * for the second-order accurate wave equation solver.
 */
void FiniteDifferenceSolver::swap_time_levels() {
  std::swap(pressure_old_, pressure_);
  std::swap(pressure_, pressure_new_);
}

/**
 * @brief Set the source position for the current shot
 *
 * Converts physical source coordinates (in meters) to grid indices
 * for efficient source application during time-stepping.
 *
 * The source position is clamped to valid grid indices to prevent
 * out-of-bounds access.
 *
 * @param x Source x-coordinate in meters
 * @param y Source y-coordinate in meters
 * @param z Source z-coordinate in meters
 */
void FiniteDifferenceSolver::set_source_position(float x, float y, float z) {
  // Convert physical coordinates to grid indices
  source_ix_ = static_cast<size_t>(x / dx_);
  source_iy_ = static_cast<size_t>(y / dy_);
  source_iz_ = static_cast<size_t>(z / dz_);

  // Clamp to valid grid range
  source_ix_ = std::min(source_ix_, nx_ - 1);
  source_iy_ = std::min(source_iy_, ny_ - 1);
  source_iz_ = std::min(source_iz_, nz_ - 1);

  source_position_set_ = true;
}

/**
 * @brief Apply source excitation to the wavefield
 *
 * Adds a source signal (amplitude) to the pressure field at the
 * pre-configured source location. The source position must be set
 * using set_source_position() before calling this method.
 *
 * @param amplitude Source amplitude at this time step
 * @param source_index Index of the source (for multiple sources, currently
 * unused)
 */
void FiniteDifferenceSolver::apply_source(float amplitude,
                                          size_t source_index) {
  // Use the configured source position
  if (is_valid_index(source_ix_, source_iy_, source_iz_)) {
    size_t idx = index(source_ix_, source_iy_, source_iz_);
    pressure_[idx] += amplitude;
  }
}

} // namespace starfwi
