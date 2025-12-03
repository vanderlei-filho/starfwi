#ifndef WAVELET_HPP
#define WAVELET_HPP

#include <cstddef>
#include <vector>

namespace starfwi {

/**
 * @brief Generate a Ricker wavelet (Mexican hat wavelet)
 *
 * The Ricker wavelet is the most commonly used source function in seismic
 * applications. It is the second derivative of a Gaussian function and
 * provides a good approximation to real seismic sources.
 *
 * Mathematical form:
 *   w(t) = (1 - 2π²f₀²t²) * exp(-π²f₀²t²)
 *
 * where f₀ is the peak frequency and t is time relative to the peak.
 *
 * Properties:
 *   - Zero DC component (integral over time = 0)
 *   - Compact support (decays rapidly)
 *   - Peak frequency occurs at f₀
 *   - Peak amplitude delayed by ~1.5/f₀ for causality
 *
 * @param frequency Peak frequency in Hz
 * @param dt Time step in seconds
 * @param nt Number of time steps to generate
 * @return Vector of wavelet amplitudes sampled at dt intervals
 */
std::vector<float> generate_ricker_wavelet(float frequency, float dt,
                                           size_t nt);

} // namespace starfwi

#endif // WAVELET_HPP
