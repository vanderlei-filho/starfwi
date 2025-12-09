#include "acoustics/misfit.hpp"
#include <cmath>
#include <stdexcept>

namespace starfwi {

MisfitCalculator::MisfitCalculator(size_t n_receivers, size_t nt)
    : n_receivers_(n_receivers), nt_(nt), misfit_(0.0f) {
  // Pre-allocate residuals array
  residuals_.resize(n_receivers_ * nt_, 0.0f);
}

float MisfitCalculator::compute(const std::vector<float> &observed_data,
                                const std::vector<float> &synthetic_data) {
  size_t expected_size = n_receivers_ * nt_;

  if (observed_data.size() != expected_size) {
    throw std::invalid_argument(
        "Observed data size mismatch: expected " + std::to_string(expected_size) +
        ", got " + std::to_string(observed_data.size()));
  }

  if (synthetic_data.size() != expected_size) {
    throw std::invalid_argument(
        "Synthetic data size mismatch: expected " + std::to_string(expected_size) +
        ", got " + std::to_string(synthetic_data.size()));
  }

  return compute(observed_data.data(), synthetic_data.data(), expected_size);
}

float MisfitCalculator::compute(const float *observed_ptr,
                                const float *synthetic_ptr, size_t size) {
  // Compute residuals and accumulate misfit
  // r = d_obs - d_syn
  // J = 0.5 × Σ r²

  float sum_squared = 0.0f;

  for (size_t i = 0; i < size; ++i) {
    float residual = observed_ptr[i] - synthetic_ptr[i];
    residuals_[i] = residual;
    sum_squared += residual * residual;
  }

  // L2 misfit: J = 0.5 × ||r||²
  misfit_ = 0.5f * sum_squared;

  return misfit_;
}

float MisfitCalculator::compute_normalized_misfit(
    const std::vector<float> &observed_data) const {
  // Compute ||d_obs||²
  float obs_energy = 0.0f;
  for (size_t i = 0; i < observed_data.size(); ++i) {
    obs_energy += observed_data[i] * observed_data[i];
  }

  // Avoid division by zero
  if (obs_energy < 1e-30f) {
    return 0.0f;
  }

  // Compute ||r||² from stored residuals
  float residual_energy = 0.0f;
  for (size_t i = 0; i < residuals_.size(); ++i) {
    residual_energy += residuals_[i] * residuals_[i];
  }

  return residual_energy / obs_energy;
}

} // namespace starfwi
