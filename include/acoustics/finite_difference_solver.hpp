#ifndef FINITE_DIFFERENCE_SOLVER_HPP
#define FINITE_DIFFERENCE_SOLVER_HPP

#include "utils/simulation_config.hpp"
#include <vector>

namespace starfwi {

// Acoustic wave equation solver using finite differences
class FiniteDifferenceSolver {
public:
  FiniteDifferenceSolver(const SimulationConfig &config);

  // Initialize wavefield and allocate memory
  // Returns false if initialization fails (e.g., velocity model size mismatch)
  bool initialize();

  // Main simulation step - advances one time step
  void step();

  // Set source position for the current shot
  void set_source_position(float x, float y, float z);

  // Apply source term at source locations
  void apply_source(float amplitude, size_t source_index);

  // Apply source at an arbitrary grid point (used to inject adjoint sources at
  // receiver positions during backward propagation)
  void apply_source_at(float amplitude, size_t ix, size_t iy, size_t iz);

  // Getters
  const std::vector<float> &get_pressure_field() const { return pressure_; }

private:
  const SimulationConfig &config_;

  // Grid dimensions
  size_t nx_, ny_, nz_;
  float dx_, dy_, dz_, dt_;

  // Wavefield arrays (3D flattened to 1D)
  // We use three time levels for the acoustic wave equation:
  // pressure_old_ (t-1), pressure_ (t), pressure_new_ (t+1)
  std::vector<float> pressure_;     // Current time step (t)
  std::vector<float> pressure_old_; // Previous time step (t-1)
  std::vector<float> pressure_new_; // Next time step (t+1)

  // Velocity model (squared for efficiency)
  std::vector<float> velocity_squared_;

  // Source position (grid indices)
  size_t source_ix_, source_iy_, source_iz_;
  bool source_position_set_;

  // Helper methods
  void initialize_fields();
  void compute_laplacian();
  void compute_laplacian_2d(); // Optimized for 2D models (ny == 1)
  void compute_laplacian_3d(); // Full 3D computation
  void apply_boundary_conditions();
  void swap_time_levels();

  // Convert 3D indices to 1D array index
  inline size_t index(size_t ix, size_t iy, size_t iz) const {
    return ix + nx_ * (iy + ny_ * iz);
  }

  // Check if index is within bounds
  inline bool is_valid_index(size_t ix, size_t iy, size_t iz) const {
    return ix < nx_ && iy < ny_ && iz < nz_;
  }
};

} // namespace starfwi

#endif // FINITE_DIFFERENCE_SOLVER_HPP
