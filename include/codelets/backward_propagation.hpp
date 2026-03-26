#ifndef BACKWARD_PROPAGATION_HPP
#define BACKWARD_PROPAGATION_HPP

#include <starpu.h>
#include <starpu_mpi.h>

namespace starfwi {

// StarPU codelet for adjoint (backward) wave propagation.
//
// Implements the adjoint state method for the acoustic wave equation:
//   1. Re-runs the forward simulation storing pressure snapshots p[t].
//   2. Propagates the adjoint field q using time-reversed residuals as sources.
//   3. Accumulates the gradient via the cross-correlation imaging condition:
//        grad[x] += (-2 / c³[x]) × p̈[x,t] × q[x,T-t] × dt
//      where p̈ = (p[t+1] - 2p[t] + p[t-1]) / dt².
//
// Buffers (same layout as forward_propagation_codelet):
//   0: velocity model  (VECTOR, STARPU_R)
//   1: shot data       (VARIABLE, STARPU_RW)  reads residuals, writes gradient
//   2: task config     (VARIABLE, STARPU_R)
//   3: receiver X      (VECTOR, STARPU_R)
//   4: receiver Y      (VECTOR, STARPU_R)
//   5: receiver Z      (VECTOR, STARPU_R)
extern struct starpu_codelet backward_propagation_codelet;

void backward_propagation_cpu(void *buffers[], void *cl_arg);

} // namespace starfwi

#endif // BACKWARD_PROPAGATION_HPP
