#ifndef FORWARD_PROPAGATION_HPP
#define FORWARD_PROPAGATION_HPP

#include <starpu.h>
#include <starpu_mpi.h>
#include <vector>

namespace starfwi {

// POD struct for task parameters (safe to transfer via MPI)
// Note: Must be POD-compatible for MPI transfer
struct TaskConfig {
  size_t nx, ny, nz;        // Grid dimensions
  float dx, dy, dz;         // Grid spacing
  float dt;                 // Time step
  size_t nt;                // Number of time steps
  size_t snapshot_interval; // Save wavefield every N timesteps (0 = disabled)
  char snapshot_dir[256];   // Output directory for snapshots

  // Global receiver array size (positions stored separately)
  size_t n_receivers;       // Number of receivers in global array

  // Directory for observed seismogram files (used by save_observed codelet)
  char observed_dir[256];

  // Forward wavefield storage strategy for backward propagation:
  //   0 = MEMORY — keep all nt snapshots in ShotData::pressure_snapshots
  //   1 = DISK   — write snapshots to a temp file in wavefield_dir
  int wavefield_storage;
  char wavefield_dir[256];

  // Number of shots assigned to this MPI rank. Used to compute the per-shot
  // RAM budget for hybrid storage so all shots fit in memory simultaneously.
  size_t shots_per_rank;
};

// Codelet argument to pass MPI rank and host information
struct CodeletArg {
  int rank;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  bool verbose;
};

// Structure to hold data for a single shot
struct ShotData {
  size_t shot_id;

  // Source information
  float source_x;
  float source_y;
  float source_z;
  std::vector<float> source_wavelet;

  // Recorded seismogram data at global receivers
  // Note: Receiver positions are defined globally, not per-shot
  std::vector<float> synthetic_data; // Synthetic seismograms: n_global_receivers × nt
  std::vector<float> observed_data;  // Observed seismograms (for FWI)

  // FWI residuals and misfit (computed after forward propagation)
  std::vector<float> residuals;      // Residuals: observed - synthetic (adjoint sources)
  float misfit = 0.0f;               // L2 misfit value for this shot

  // Gradient contribution from this shot (same size as velocity model)
  // Accumulated via cross-correlation imaging condition during backward propagation.
  // Must be summed across all shots (MPI reduction) to get the full gradient.
  std::vector<float> gradient;

  // Forward pressure snapshots — populated by forward_propagation (MEMORY strategy)
  // and consumed + cleared by backward_propagation. Layout: [t * grid_size + i].
  std::vector<float> pressure_snapshots;

  // Actual snapshot storage mode chosen at runtime by the forward codelet.
  // May differ from task_config->wavefield_storage when auto-detection overrides
  // the requested mode based on available GPU/host memory.
  //  -1 = NONE    (modeling path, no backward needed)
  //   0 = MEMORY  (all nt snapshots in pressure_snapshots)
  //   1 = DISK    (all nt snapshots in a binary file)
  //   2 = HYBRID  (first snapshots_in_ram snapshots in pressure_snapshots,
  //                remaining nt - snapshots_in_ram snapshots in a binary file)
  int wavefield_storage_actual = -1;
  size_t snapshots_in_ram = 0; // HYBRID only: number of snapshots kept in RAM
};

// StarPU codelet for forward wave propagation
extern struct starpu_codelet forward_propagation_codelet;

// CPU implementation
void forward_propagation_cpu(void *buffers[], void *cl_arg);

// CUDA implementation — only available when built with CUDA support
#ifdef STARPU_USE_CUDA
void forward_propagation_cuda(void *buffers[], void *cl_arg);
#endif

} // namespace starfwi

#endif // FORWARD_PROPAGATION_HPP
