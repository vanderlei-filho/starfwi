#ifndef COMPUTE_MISFIT_HPP
#define COMPUTE_MISFIT_HPP

#include <starpu.h>

namespace starfwi {

// StarPU codelet for computing misfit and residuals
extern struct starpu_codelet compute_misfit_codelet;

// CPU implementation
void compute_misfit_cpu(void *buffers[], void *cl_arg);

} // namespace starfwi

#endif // COMPUTE_MISFIT_HPP
