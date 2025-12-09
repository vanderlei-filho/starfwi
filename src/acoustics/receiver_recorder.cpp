#include "acoustics/receiver_recorder.hpp"
#include "acoustics/finite_difference_solver.hpp"
#include <algorithm>
#include <cmath>

namespace starfwi {

ReceiverRecorder::ReceiverRecorder(const float *receiver_x_ptr,
                                   const float *receiver_y_ptr,
                                   const float *receiver_z_ptr,
                                   size_t n_receivers, size_t nt, size_t nx,
                                   size_t ny, size_t nz, float dx, float dy,
                                   float dz)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz), n_receivers_(n_receivers),
      nt_(nt) {

  // Copy receiver positions from input pointers
  receiver_x_.assign(receiver_x_ptr, receiver_x_ptr + n_receivers);
  receiver_y_.assign(receiver_y_ptr, receiver_y_ptr + n_receivers);
  receiver_z_.assign(receiver_z_ptr, receiver_z_ptr + n_receivers);

  // Allocate storage for synthetic data: n_receivers × nt
  synthetic_data_.resize(n_receivers_ * nt, 0.0f);
}

void ReceiverRecorder::record_timestep(const FiniteDifferenceSolver &propagator,
                                       size_t timestep) {
  // Get pressure field from propagator
  const std::vector<float> &pressure_field = propagator.get_pressure_field();

  // Record at each receiver location
  for (size_t ir = 0; ir < n_receivers_; ++ir) {
    float x = receiver_x_[ir];
    float y = receiver_y_[ir];
    float z = receiver_z_[ir];

    // Interpolate pressure at receiver position
    float value = interpolate_at_point(pressure_field, x, y, z);

    // Store in synthetic data array: [receiver][time]
    synthetic_data_[ir * nt_ + timestep] = value;
  }
}

float ReceiverRecorder::interpolate_at_point(
    const std::vector<float> &pressure_field, float x, float y, float z) const {

  // Convert physical coordinates to grid coordinates
  float fx = x / dx_;
  float fy = y / dy_;
  float fz = z / dz_;

  // Get integer indices (floor)
  size_t ix0 = static_cast<size_t>(std::floor(fx));
  size_t iy0 = static_cast<size_t>(std::floor(fy));
  size_t iz0 = static_cast<size_t>(std::floor(fz));

  // Get fractional parts for interpolation weights
  float wx = fx - ix0;
  float wy = fy - iy0;
  float wz = fz - iz0;

  // Clamp indices to valid range
  size_t ix1 = std::min(ix0 + 1, nx_ - 1);
  size_t iy1 = std::min(iy0 + 1, ny_ - 1);
  size_t iz1 = std::min(iz0 + 1, nz_ - 1);

  ix0 = std::min(ix0, nx_ - 1);
  iy0 = std::min(iy0, ny_ - 1);
  iz0 = std::min(iz0, nz_ - 1);

  // Trilinear interpolation
  // Get values at 8 surrounding grid points
  float v000 = pressure_field[index(ix0, iy0, iz0)];
  float v100 = pressure_field[index(ix1, iy0, iz0)];
  float v010 = pressure_field[index(ix0, iy1, iz0)];
  float v110 = pressure_field[index(ix1, iy1, iz0)];
  float v001 = pressure_field[index(ix0, iy0, iz1)];
  float v101 = pressure_field[index(ix1, iy0, iz1)];
  float v011 = pressure_field[index(ix0, iy1, iz1)];
  float v111 = pressure_field[index(ix1, iy1, iz1)];

  // Interpolate along x
  float v00 = v000 * (1.0f - wx) + v100 * wx;
  float v01 = v001 * (1.0f - wx) + v101 * wx;
  float v10 = v010 * (1.0f - wx) + v110 * wx;
  float v11 = v011 * (1.0f - wx) + v111 * wx;

  // Interpolate along y
  float v0 = v00 * (1.0f - wy) + v10 * wy;
  float v1 = v01 * (1.0f - wy) + v11 * wy;

  // Interpolate along z
  float value = v0 * (1.0f - wz) + v1 * wz;

  return value;
}

} // namespace starfwi
