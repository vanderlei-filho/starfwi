#include "utils/snapshot_writer.hpp"
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>

namespace starfwi {
namespace utils {

bool SnapshotWriter::write_snapshot(const std::string &filename,
                                    const std::vector<float> &data, uint32_t nx,
                                    uint32_t ny, uint32_t nz, float dx,
                                    float dy, float dz, uint32_t timestep,
                                    float dt, FieldType field_type) {
  // Validate input
  size_t expected_size = static_cast<size_t>(nx) * ny * nz;
  if (data.size() != expected_size) {
    std::println(stderr,
                 "[snapshot_writer] ERROR: Data size mismatch. Expected {} "
                 "elements, got {}",
                 expected_size, data.size());
    return false;
  }

  // Create output directory if it doesn't exist
  std::filesystem::path filepath(filename);
  if (filepath.has_parent_path()) {
    std::filesystem::create_directories(filepath.parent_path());
  }

  // Open file for binary writing
  std::ofstream file(filename, std::ios::binary | std::ios::trunc);
  if (!file) {
    std::println(stderr, "[snapshot_writer] ERROR: Failed to open file: {}",
                 filename);
    return false;
  }

  // Write header
  // Magic number "SWAV"
  file.write(MAGIC, 4);

  // Version
  file.write(reinterpret_cast<const char *>(&FORMAT_VERSION),
             sizeof(FORMAT_VERSION));

  // Grid dimensions
  file.write(reinterpret_cast<const char *>(&nx), sizeof(nx));
  file.write(reinterpret_cast<const char *>(&ny), sizeof(ny));
  file.write(reinterpret_cast<const char *>(&nz), sizeof(nz));

  // Grid spacing
  file.write(reinterpret_cast<const char *>(&dx), sizeof(dx));
  file.write(reinterpret_cast<const char *>(&dy), sizeof(dy));
  file.write(reinterpret_cast<const char *>(&dz), sizeof(dz));

  // Time info
  file.write(reinterpret_cast<const char *>(&timestep), sizeof(timestep));
  file.write(reinterpret_cast<const char *>(&dt), sizeof(dt));

  // Field type
  uint32_t field_type_value = static_cast<uint32_t>(field_type);
  file.write(reinterpret_cast<const char *>(&field_type_value),
             sizeof(field_type_value));

  // Write data array
  file.write(reinterpret_cast<const char *>(data.data()),
             data.size() * sizeof(float));

  if (!file) {
    std::println(stderr, "[snapshot_writer] ERROR: Failed to write data to {}",
                 filename);
    return false;
  }

  return true;
}

std::string SnapshotWriter::generate_filename(const std::string &base_dir,
                                              FieldType field_type,
                                              uint32_t shot_id,
                                              uint32_t timestep) {
  // Determine field name
  std::string field_name;
  switch (field_type) {
  case FieldType::VELOCITY:
    field_name = "velocity";
    break;
  case FieldType::PRESSURE:
    field_name = "pressure";
    break;
  case FieldType::GRADIENT:
    field_name = "gradient";
    break;
  case FieldType::ADJOINT:
    field_name = "adjoint";
    break;
  case FieldType::RESIDUAL:
    field_name = "residual";
    break;
  default:
    field_name = "unknown";
  }

  // For velocity model (shot_id = 0), don't create shot subdirectory
  if (field_type == FieldType::VELOCITY) {
    return std::format("{}/velocity_model.bin", base_dir);
  }

  // For shot-specific data, create shot subdirectory
  return std::format("{}/shot_{:03d}/{}_{:04d}.bin", base_dir, shot_id,
                     field_name, timestep);
}

} // namespace utils
} // namespace starfwi
