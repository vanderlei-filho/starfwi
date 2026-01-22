#include "codelets/save_observed.hpp"
#include "codelets/forward_propagation.hpp" // For ShotData, TaskConfig, CodeletArg
#include "utils/seismogram_io.hpp"
#include <print>

namespace starfwi {

/**
 * @brief StarPU codelet for saving observed seismogram data
 *
 * This codelet saves synthetic seismograms as "observed" data files.
 * Used when generating observed data from the true velocity model.
 *
 * Buffers:
 *   - Buffer 0: Shot data (VARIABLE, read-write, STARPU_RW)
 *   - Buffer 1: Task config (VARIABLE, read-only, STARPU_R)
 *
 * Note: Shot data is STARPU_RW (not STARPU_R) because StarPU-MPI requires
 * at least one RW buffer to determine task ownership/placement. The data
 * is not actually modified, but this ensures correct distributed execution.
 */
struct starpu_codelet save_observed_codelet = {
    .cpu_funcs = {save_observed_cpu},
    .cpu_funcs_name = {"save_observed_cpu"},
    .nbuffers = 2,
    .modes = {STARPU_RW, STARPU_R},
    .name = "save_observed",
    .color = 0x00FF00};

void save_observed_cpu(void *buffers[], void *cl_arg) {
  ShotData *shot = (ShotData *)STARPU_VARIABLE_GET_PTR(buffers[0]);
  TaskConfig *task_config = (TaskConfig *)STARPU_VARIABLE_GET_PTR(buffers[1]);

  CodeletArg *arg = (CodeletArg *)cl_arg;
  const char *hostname = arg ? arg->hostname : "unknown";
  bool verbose = arg ? arg->verbose : false;

  if (shot->synthetic_data.empty()) {
    if (verbose) {
      std::println("[starfwi][{}][save_observed_cpu] Shot {} has no synthetic "
                   "data to save",
                   hostname, shot->shot_id);
    }
    return;
  }

  utils::SeismogramIO::Header header;
  header.shot_id = shot->shot_id;
  header.n_receivers = task_config->n_receivers;
  header.nt = task_config->nt;
  header.dt = task_config->dt;
  header.source_x = shot->source_x;
  header.source_y = shot->source_y;
  header.source_z = shot->source_z;

  std::string filename =
      utils::SeismogramIO::generate_filename(task_config->observed_dir,
                                             shot->shot_id);

  auto result = utils::SeismogramIO::save(filename, shot->synthetic_data, header);

  if (result) {
    if (verbose) {
      std::println("[starfwi][{}][save_observed_cpu] Saved observed data for "
                   "shot {} to {}",
                   hostname, shot->shot_id, filename);
    }
  } else {
    std::println(stderr,
                 "[starfwi][{}][save_observed_cpu] ERROR: Failed to save "
                 "observed data for shot {}: {}",
                 hostname, shot->shot_id, result.error());
  }
}

} // namespace starfwi
