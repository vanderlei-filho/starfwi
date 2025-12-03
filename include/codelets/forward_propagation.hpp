#ifndef FORWARD_PROPAGATION_HPP
#define FORWARD_PROPAGATION_HPP

#include <starpu.h>
#include <starpu_mpi.h>
#include <vector>

namespace starfwi {

// POD struct for task parameters (safe to transfer via MPI)
struct TaskConfig {
  size_t nx, ny, nz;        // Grid dimensions
  float dx, dy, dz;         // Grid spacing
  float dt;                 // Time step
  size_t nt;                // Number of time steps
  size_t snapshot_interval; // Save wavefield every N timesteps (0 = disabled)
  char snapshot_dir[256];   // Output directory for snapshots
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
  float source_x;
  float source_y;
  float source_z;
  std::vector<float> source_wavelet;
  std::vector<float> receiver_data; // Output: recorded seismograms
};

// StarPU codelet for forward wave propagation
extern struct starpu_codelet forward_propagation_codelet;

// The actual computation function called by StarPU
void forward_propagation_cpu(void *buffers[], void *cl_arg);

} // namespace starfwi

#endif // FORWARD_PROPAGATION_HPP
