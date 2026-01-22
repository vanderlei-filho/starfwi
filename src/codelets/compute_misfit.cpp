#include "codelets/compute_misfit.hpp"
#include "codelets/forward_propagation.hpp" // For ShotData, TaskConfig, CodeletArg
#include "acoustics/misfit.hpp"
#include <print>

namespace starfwi {

/**
 * @brief StarPU codelet for computing misfit and residuals
 *
 * This codelet computes the L2 misfit between observed and synthetic data,
 * and stores the residuals for use as adjoint sources.
 *
 * Buffers:
 *   - Buffer 0: Shot data (VARIABLE, read-write, STARPU_RW)
 *   - Buffer 1: Task config (VARIABLE, read-only, STARPU_R)
 *
 * The shot data is read-write because we store the computed residuals
 * and misfit value back into the shot structure.
 */
struct starpu_codelet compute_misfit_codelet = {
    .cpu_funcs = {compute_misfit_cpu},
    .cpu_funcs_name = {"compute_misfit_cpu"},
    .nbuffers = 2,
    .modes = {STARPU_RW, STARPU_R},
    .name = "compute_misfit",
    .color = 0x0000FF};

void compute_misfit_cpu(void *buffers[], void *cl_arg) {
  ShotData *shot = (ShotData *)STARPU_VARIABLE_GET_PTR(buffers[0]);
  TaskConfig *task_config = (TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[1]);

  CodeletArg *arg = (CodeletArg *)cl_arg;
  const char *hostname = arg ? arg->hostname : "unknown";
  bool verbose = arg ? arg->verbose : false;

  // Check if we have both observed and synthetic data
  if (shot->observed_data.empty()) {
    if (verbose) {
      std::println("[starfwi][{}][compute_misfit_cpu] Shot {} has no observed "
                   "data, skipping misfit computation",
                   hostname, shot->shot_id);
    }
    return;
  }

  if (shot->synthetic_data.empty()) {
    if (verbose) {
      std::println("[starfwi][{}][compute_misfit_cpu] Shot {} has no synthetic "
                   "data, skipping misfit computation",
                   hostname, shot->shot_id);
    }
    return;
  }

  // Compute misfit and residuals
  MisfitCalculator misfit_calc(task_config->n_receivers, task_config->nt);
  shot->misfit = misfit_calc.compute(shot->observed_data, shot->synthetic_data);
  shot->residuals = misfit_calc.get_residuals();

  if (verbose) {
    float normalized =
        misfit_calc.compute_normalized_misfit(shot->observed_data);
    std::println("[starfwi][{}][compute_misfit_cpu] Shot {} misfit: {:.6e} "
                 "(normalized: {:.4f})",
                 hostname, shot->shot_id, shot->misfit, normalized);
  }
}

} // namespace starfwi
