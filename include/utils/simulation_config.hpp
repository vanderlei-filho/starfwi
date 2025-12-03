#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace starfwi {

// Structure to hold 3D grid dimensions
struct GridDimensions {
  size_t nx = 0;   // Number of grid points in x direction
  size_t ny = 0;   // Number of grid points in y direction
  size_t nz = 0;   // Number of grid points in z direction
  float dx = 0.0f; // Grid spacing in x direction (meters)
  float dy = 0.0f; // Grid spacing in y direction (meters)
  float dz = 0.0f; // Grid spacing in z direction (meters)
};

// Structure to hold time stepping parameters
struct TimeParameters {
  float dt = 0.0f;    // Time step (seconds)
  size_t nt = 0;      // Number of time steps
  float t_max = 0.0f; // Maximum simulation time (seconds)
};

// Structure for source configuration
struct SourceConfig {
  std::vector<float> x_positions; // X coordinates of sources
  std::vector<float> y_positions; // Y coordinates of sources
  std::vector<float> z_positions; // Z coordinates of sources
  float frequency = 0.0f;         // Source frequency (Hz)
};

// Structure for receiver configuration
struct ReceiverConfig {
  std::vector<float> x_positions; // X coordinates of receivers
  std::vector<float> y_positions; // Y coordinates of receivers
  std::vector<float> z_positions; // Z coordinates of receivers
  float spacing = 0.0f;           // Receiver spacing (meters)
};

// Structure to hold velocity model information
struct VelocityModel {
  std::string filepath;    // Path to SEG-Y file
  std::vector<float> data; // Velocity data (m/s)
  bool is_loaded;          // Flag indicating if model is loaded
  VelocityModel() : is_loaded(false) {}
};

// Main input configuration structure
struct SimulationConfig {
  GridDimensions grid;
  TimeParameters time;
  SourceConfig source;
  ReceiverConfig receiver;
  VelocityModel velocity_model;

  // Output settings
  std::string output_directory;
  bool save_wavefield;
  size_t snapshot_interval;

  // Computational settings
  size_t num_workers; // Number of StarPU workers
  bool use_mpi;       // Enable MPI parallelization

  SimulationConfig()
      : save_wavefield(false), snapshot_interval(10), num_workers(0),
        use_mpi(false) {}
};

} // namespace starfwi

#endif // SIMULATION_CONFIG_HPP
