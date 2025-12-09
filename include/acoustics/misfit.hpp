#ifndef MISFIT_HPP
#define MISFIT_HPP

#include <cstddef>
#include <vector>

namespace starfwi {

/**
 * @brief Computes misfit (objective function) and residuals for FWI
 *
 * This class handles the computation of:
 * - Data residuals: r = d_obs - d_syn (observed - synthetic)
 * - L2 misfit: J = 0.5 × ||d_obs - d_syn||²
 *
 * The residuals are used as adjoint sources for backpropagation.
 * The misfit value is used to monitor convergence of the inversion.
 *
 * Data layout: [receiver][time] in row-major order (n_receivers × nt)
 */
class MisfitCalculator {
public:
  /**
   * @brief Construct a misfit calculator
   *
   * @param n_receivers Number of receivers
   * @param nt Number of timesteps
   */
  MisfitCalculator(size_t n_receivers, size_t nt);

  /**
   * @brief Compute residuals and L2 misfit
   *
   * Computes:
   * - Residuals: r[i][t] = observed[i][t] - synthetic[i][t]
   * - L2 misfit: J = 0.5 × Σ r[i][t]²
   *
   * The residuals are stored internally and can be retrieved with
   * get_residuals(). These residuals serve as adjoint sources.
   *
   * @param observed_data Observed seismograms [n_receivers × nt]
   * @param synthetic_data Synthetic seismograms [n_receivers × nt]
   * @return L2 misfit value (scalar)
   */
  float compute(const std::vector<float> &observed_data,
                const std::vector<float> &synthetic_data);

  /**
   * @brief Compute residuals and L2 misfit from raw pointers
   *
   * @param observed_ptr Pointer to observed data
   * @param synthetic_ptr Pointer to synthetic data
   * @param size Total number of elements (n_receivers × nt)
   * @return L2 misfit value (scalar)
   */
  float compute(const float *observed_ptr, const float *synthetic_ptr,
                size_t size);

  /**
   * @brief Get the computed residuals (adjoint sources)
   *
   * The residuals are: r = observed - synthetic
   * These are injected at receiver locations during adjoint propagation.
   *
   * Data layout: [receiver][time] in row-major order
   *
   * @return Const reference to residuals vector
   */
  const std::vector<float> &get_residuals() const { return residuals_; }

  /**
   * @brief Get mutable reference to residuals for direct manipulation
   *
   * Useful for applying preprocessing (e.g., muting, filtering) to residuals
   * before using them as adjoint sources.
   *
   * @return Mutable reference to residuals vector
   */
  std::vector<float> &get_residuals() { return residuals_; }

  /**
   * @brief Get the last computed misfit value
   *
   * @return L2 misfit value from the most recent compute() call
   */
  float get_misfit() const { return misfit_; }

  /**
   * @brief Get number of receivers
   */
  size_t get_num_receivers() const { return n_receivers_; }

  /**
   * @brief Get number of timesteps
   */
  size_t get_num_timesteps() const { return nt_; }

  /**
   * @brief Get residual value at specific receiver and timestep
   *
   * @param receiver_idx Receiver index (0 to n_receivers-1)
   * @param timestep Timestep index (0 to nt-1)
   * @return Residual value
   */
  float get_residual_at(size_t receiver_idx, size_t timestep) const {
    return residuals_[receiver_idx * nt_ + timestep];
  }

  /**
   * @brief Compute normalized misfit (useful for convergence monitoring)
   *
   * Returns misfit normalized by the energy of observed data:
   * J_norm = ||d_obs - d_syn||² / ||d_obs||²
   *
   * @param observed_data Observed seismograms
   * @return Normalized misfit value (0 = perfect fit, 1 = zero correlation)
   */
  float
  compute_normalized_misfit(const std::vector<float> &observed_data) const;

private:
  size_t n_receivers_;
  size_t nt_;

  // Residuals: observed - synthetic [n_receivers × nt]
  std::vector<float> residuals_;

  // Last computed misfit value
  float misfit_;
};

} // namespace starfwi

#endif // MISFIT_HPP
