#ifndef SAVE_OBSERVED_HPP
#define SAVE_OBSERVED_HPP

#include <starpu.h>

namespace starfwi {

// StarPU codelet for saving observed seismogram data
extern struct starpu_codelet save_observed_codelet;

// CPU implementation
void save_observed_cpu(void *buffers[], void *cl_arg);

} // namespace starfwi

#endif // SAVE_OBSERVED_HPP
