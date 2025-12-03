#include "utils/wavelet.hpp"
#include <cmath>

// Define M_PI if not already defined (some compilers don't define it)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace starfwi {

std::vector<float> generate_ricker_wavelet(float frequency, float dt,
                                           size_t nt) {
  std::vector<float> wavelet(nt, 0.0f);

  // Peak frequency parameter
  const float f0 = frequency;
  const float omega = M_PI * f0;

  // Time delay to ensure causality (peak occurs at ~1.5/f0)
  const float t_delay = 1.5f / f0;

  // Generate the Ricker wavelet
  for (size_t i = 0; i < nt; ++i) {
    float t = i * dt - t_delay;
    float arg = omega * omega * t * t;

    // Ricker wavelet formula: (1 - 2*arg) * exp(-arg)
    wavelet[i] = (1.0f - 2.0f * arg) * std::exp(-arg);
  }

  return wavelet;
}

} // namespace starfwi
