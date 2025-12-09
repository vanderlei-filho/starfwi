#include "utils/seismogram_io.hpp"
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>

namespace starfwi {
namespace utils {

std::expected<void, std::string>
SeismogramIO::save(const std::string &filename, const std::vector<float> &data,
                   const Header &header) {
  // Validate data size
  size_t expected_size =
      static_cast<size_t>(header.n_receivers) * static_cast<size_t>(header.nt);
  if (data.size() != expected_size) {
    return std::unexpected(
        std::format("Data size mismatch: expected {} ({}×{}), got {}",
                    expected_size, header.n_receivers, header.nt, data.size()));
  }

  // Ensure output directory exists
  std::filesystem::path filepath(filename);
  if (filepath.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(filepath.parent_path(), ec);
    if (ec) {
      return std::unexpected(
          std::format("Failed to create directory: {}", ec.message()));
    }
  }

  // Open file for binary writing
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    return std::unexpected(
        std::format("Failed to open file for writing: {}", filename));
  }

  // Write magic number
  file.write(MAGIC, 4);

  // Write version
  file.write(reinterpret_cast<const char *>(&FORMAT_VERSION), sizeof(uint32_t));

  // Write header fields
  file.write(reinterpret_cast<const char *>(&header.shot_id), sizeof(uint32_t));
  file.write(reinterpret_cast<const char *>(&header.n_receivers),
             sizeof(uint32_t));
  file.write(reinterpret_cast<const char *>(&header.nt), sizeof(uint32_t));
  file.write(reinterpret_cast<const char *>(&header.dt), sizeof(float));
  file.write(reinterpret_cast<const char *>(&header.source_x), sizeof(float));
  file.write(reinterpret_cast<const char *>(&header.source_y), sizeof(float));
  file.write(reinterpret_cast<const char *>(&header.source_z), sizeof(float));

  // Write reserved bytes (32 bytes of zeros for future use)
  char reserved[32] = {0};
  file.write(reserved, 32);

  // Write seismogram data
  file.write(reinterpret_cast<const char *>(data.data()),
             data.size() * sizeof(float));

  if (!file) {
    return std::unexpected("Failed to write seismogram data");
  }

  return {};
}

std::expected<void, std::string> SeismogramIO::load(const std::string &filename,
                                                    std::vector<float> &data,
                                                    Header &header) {
  // First read the header
  auto header_result = read_header(filename, header);
  if (!header_result) {
    return std::unexpected(header_result.error());
  }

  // Open file and seek past header
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return std::unexpected(
        std::format("Failed to open file for reading: {}", filename));
  }

  file.seekg(HEADER_SIZE);

  // Calculate data size and allocate
  size_t data_size =
      static_cast<size_t>(header.n_receivers) * static_cast<size_t>(header.nt);
  data.resize(data_size);

  // Read seismogram data
  file.read(reinterpret_cast<char *>(data.data()), data_size * sizeof(float));

  if (!file) {
    return std::unexpected("Failed to read seismogram data");
  }

  return {};
}

std::expected<void, std::string>
SeismogramIO::read_header(const std::string &filename, Header &header) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("Failed to open file: {}", filename));
  }

  // Read and verify magic number
  char magic[4];
  file.read(magic, 4);
  if (std::memcmp(magic, MAGIC, 4) != 0) {
    return std::unexpected("Invalid file format: magic number mismatch");
  }

  // Read and verify version
  uint32_t version;
  file.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
  if (version != FORMAT_VERSION) {
    return std::unexpected(std::format(
        "Unsupported file version: {} (expected {})", version, FORMAT_VERSION));
  }

  // Read header fields
  file.read(reinterpret_cast<char *>(&header.shot_id), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&header.n_receivers), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&header.nt), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&header.dt), sizeof(float));
  file.read(reinterpret_cast<char *>(&header.source_x), sizeof(float));
  file.read(reinterpret_cast<char *>(&header.source_y), sizeof(float));
  file.read(reinterpret_cast<char *>(&header.source_z), sizeof(float));

  // Skip reserved bytes
  file.seekg(32, std::ios::cur);

  if (!file) {
    return std::unexpected("Failed to read header");
  }

  return {};
}

std::string SeismogramIO::generate_filename(const std::string &base_dir,
                                            uint32_t shot_id) {
  return std::format("{}/observed_shot_{:03d}.bin", base_dir, shot_id);
}

} // namespace utils
} // namespace starfwi
