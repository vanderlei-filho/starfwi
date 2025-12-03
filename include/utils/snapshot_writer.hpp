#ifndef SNAPSHOT_WRITER_HPP
#define SNAPSHOT_WRITER_HPP

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace starfwi {
namespace utils {

// Field types for different simulation data
enum class FieldType : uint32_t {
  VELOCITY = 0, // Velocity model
  PRESSURE = 1, // Forward wavefield (pressure)
  GRADIENT = 2, // FWI gradient
  ADJOINT = 3,  // Adjoint wavefield
  RESIDUAL = 4  // Data residuals
};

// Snapshot metadata and writer
class SnapshotWriter {
public:
  // Write a 3D field to binary file with metadata header
  // filename: output file path (will overwrite if exists)
  // data: flattened 3D array (nx × ny × nz)
  // nx, ny, nz: grid dimensions
  // dx, dy, dz: grid spacing in meters
  // timestep: current timestep index
  // dt: timestep size in seconds
  // field_type: type of field being saved
  // Returns: void on success, error message on failure
  static std::expected<void, std::string>
  write_snapshot(const std::string &filename, const std::vector<float> &data,
                 uint32_t nx, uint32_t ny, uint32_t nz, float dx, float dy,
                 float dz, uint32_t timestep, float dt, FieldType field_type);

  // Helper to generate standard filename
  // base_dir: output directory
  // field_type: type of field
  // shot_id: shot number (0 for velocity model)
  // timestep: timestep index
  // Returns: formatted path like "output/shot_001/pressure_t0042.bin"
  static std::string generate_filename(const std::string &base_dir,
                                       FieldType field_type, uint32_t shot_id,
                                       uint32_t timestep);

private:
  // Binary file format version
  static constexpr uint32_t FORMAT_VERSION = 1;

  // Magic number to identify file type: "SWAV" (Starfwi WAVefield)
  static constexpr char MAGIC[4] = {'S', 'W', 'A', 'V'};
};

} // namespace utils
} // namespace starfwi

#endif // SNAPSHOT_WRITER_HPP
