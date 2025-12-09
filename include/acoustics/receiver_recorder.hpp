#ifndef RECEIVER_RECORDER_HPP
#define RECEIVER_RECORDER_HPP

#include <cstddef>
#include <vector>

namespace starfwi {

// Forward declaration
class FiniteDifferenceSolver;

/**
 * @brief Handles recording of seismic traces at receiver locations
 *
 * This class manages receiver geometry and records synthetic seismograms
 * during forward wave propagation. At each timestep, it interpolates
 * the pressure field at receiver positions and stores the values.
 *
 * The recorded data (synthetic seismograms) can be used for:
 * - Comparison with observed data
 * - Computing residuals for FWI
 * - Visualization of seismic traces
 */
class ReceiverRecorder {
public:
  /**
   * @brief Construct a recorder for given receiver geometry
   *
   * @param receiver_x_ptr Pointer to X coordinates of receivers (meters)
   * @param receiver_y_ptr Pointer to Y coordinates of receivers (meters)
   * @param receiver_z_ptr Pointer to Z coordinates of receivers (meters)
   * @param n_receivers Number of receivers
   * @param nt Number of timesteps to record
   * @param nx Grid dimension in X
   * @param ny Grid dimension in Y
   * @param nz Grid dimension in Z
   * @param dx Grid spacing in X (meters)
   * @param dy Grid spacing in Y (meters)
   * @param dz Grid spacing in Z (meters)
   */
  ReceiverRecorder(const float *receiver_x_ptr, const float *receiver_y_ptr,
                   const float *receiver_z_ptr, size_t n_receivers, size_t nt,
                   size_t nx, size_t ny, size_t nz, float dx, float dy,
                   float dz);

  /**
   * @brief Record pressure values at all receivers for current timestep
   *
   * Interpolates the pressure field at each receiver location and stores
   * the values in the synthetic data array.
   *
   * @param propagator Finite difference solver containing current wavefield
   * @param timestep Current timestep index
   */
  void record_timestep(const FiniteDifferenceSolver &propagator,
                       size_t timestep);

  /**
   * @brief Get the recorded synthetic seismogram data
   *
   * Data is stored in row-major format: [receiver][time]
   * Size: n_receivers × nt
   *
   * @return Const reference to synthetic data vector
   */
  const std::vector<float> &get_synthetic_data() const {
    return synthetic_data_;
  }

  /**
   * @brief Get number of receivers
   */
  size_t get_num_receivers() const { return n_receivers_; }

  /**
   * @brief Get number of timesteps
   */
  size_t get_num_timesteps() const { return nt_; }

private:
  // Receiver positions (physical coordinates in meters)
  std::vector<float> receiver_x_;
  std::vector<float> receiver_y_;
  std::vector<float> receiver_z_;

  // Grid parameters for coordinate conversion
  size_t nx_, ny_, nz_;
  float dx_, dy_, dz_;

  // Recording parameters
  size_t n_receivers_;
  size_t nt_;

  // Recorded data: [receiver][time] in row-major order
  // Total size: n_receivers × nt
  std::vector<float> synthetic_data_;

  /**
   * @brief Interpolate pressure field at a point using trilinear interpolation
   *
   * @param pressure_field The 3D pressure field (flattened)
   * @param x X coordinate in meters
   * @param y Y coordinate in meters
   * @param z Z coordinate in meters
   * @return Interpolated pressure value
   */
  float interpolate_at_point(const std::vector<float> &pressure_field, float x,
                             float y, float z) const;

  /**
   * @brief Convert 3D grid indices to 1D array index
   */
  inline size_t index(size_t ix, size_t iy, size_t iz) const {
    return ix + nx_ * (iy + ny_ * iz);
  }
};

} // namespace starfwi

#endif // RECEIVER_RECORDER_HPP
