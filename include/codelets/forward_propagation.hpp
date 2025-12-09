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

  // FWI options
  bool generate_observed;   // If true, save synthetic as observed data
  char observed_dir[256];   // Directory for observed data
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
};

// StarPU codelet for forward wave propagation
extern struct starpu_codelet forward_propagation_codelet;

// The actual computation function called by StarPU
void forward_propagation_cpu(void *buffers[], void *cl_arg);

} // namespace starfwi

#endif // FORWARD_PROPAGATION_HPP
