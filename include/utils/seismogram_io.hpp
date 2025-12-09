#ifndef SEISMOGRAM_IO_HPP
#define SEISMOGRAM_IO_HPP

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace starfwi {
namespace utils {

/**
 * @brief Seismogram I/O for saving and loading receiver data
 *
 * This class handles serialization of seismogram data (receiver recordings)
 * to binary files. Used for:
 * - Saving "observed" data generated from true model
 * - Loading observed data for FWI inversion
 * - Archiving synthetic seismograms
 *
 * File format:
 *   - Magic number: "SSGY" (4 bytes)
 *   - Version: uint32_t (4 bytes)
 *   - Shot ID: uint32_t (4 bytes)
 *   - Number of receivers: uint32_t (4 bytes)
 *   - Number of timesteps: uint32_t (4 bytes)
 *   - Time step dt: float (4 bytes)
 *   - Source position X: float (4 bytes)
 *   - Source position Y: float (4 bytes)
 *   - Source position Z: float (4 bytes)
 *   - Reserved: 32 bytes (for future metadata)
 *   - Data: float[n_receivers × nt] (row-major: [receiver][time])
 *
 * Total header size: 68 bytes
 */
class SeismogramIO {
public:
  /**
   * @brief Metadata header for seismogram files
   */
  struct Header {
    uint32_t shot_id;
    uint32_t n_receivers;
    uint32_t nt;
    float dt;
    float source_x;
    float source_y;
    float source_z;
  };

  /**
   * @brief Save seismogram data to binary file
   *
   * @param filename Output file path
   * @param data Seismogram data [n_receivers × nt] in row-major order
   * @param header Metadata header
   * @return void on success, error message on failure
   */
  static std::expected<void, std::string>
  save(const std::string &filename, const std::vector<float> &data,
       const Header &header);

  /**
   * @brief Load seismogram data from binary file
   *
   * @param filename Input file path
   * @param data Output vector to store loaded data
   * @param header Output header to store metadata
   * @return void on success, error message on failure
   */
  static std::expected<void, std::string> load(const std::string &filename,
                                               std::vector<float> &data,
                                               Header &header);

  /**
   * @brief Read only the header from a seismogram file
   *
   * Useful for checking metadata without loading all data.
   *
   * @param filename Input file path
   * @param header Output header to store metadata
   * @return void on success, error message on failure
   */
  static std::expected<void, std::string>
  read_header(const std::string &filename, Header &header);

  /**
   * @brief Generate standard filename for observed seismogram files
   *
   * @param base_dir Output directory
   * @param shot_id Shot number
   * @return Formatted path like "output/observed_shot_001.bin"
   */
  static std::string generate_filename(const std::string &base_dir,
                                       uint32_t shot_id);

private:
  // Binary file format version
  static constexpr uint32_t FORMAT_VERSION = 1;

  // Magic number: "SSGY" (Starfwi SeimoGram Y)
  static constexpr char MAGIC[4] = {'S', 'S', 'G', 'Y'};

  // Header size in bytes (fixed for forward compatibility)
  static constexpr size_t HEADER_SIZE = 68;
};

} // namespace utils
} // namespace starfwi

#endif // SEISMOGRAM_IO_HPP
