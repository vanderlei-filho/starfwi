#ifndef SEGY_READER_HPP
#define SEGY_READER_HPP

#include "simulation_config.hpp"
#include <expected>
#include <string>

namespace starfwi {

/**
 * @brief Class for reading and parsing SEG-Y velocity model files
 *
 * Supports reading SEG-Y format files with both IBM and IEEE float formats.
 * Automatically infers grid parameters from file structure.
 */
class SEGYLoader {
public:
  /**
   * @brief Load a SEG-Y velocity model from file
   *
   * Automatically detects data format (IBM/IEEE float) and handles
   * endianness conversion.
   *
   * @param filepath Path to SEG-Y file
   * @param verbose If true, print progress messages during loading
   * @return Expected with void on success, or error message string on failure
   */
  std::expected<void, std::string> load_segy_model(const std::string &filepath,
                                                   bool verbose = true);

  /**
   * @brief Get the loaded configuration
   * @return Const reference to SimulationConfig
   */
  const SimulationConfig &get_config() const { return config_; }

  /**
   * @brief Get mutable configuration
   * @return Mutable reference to SimulationConfig
   */
  SimulationConfig &get_config() { return config_; }

private:
  SimulationConfig config_;
};

} // namespace starfwi

#endif // SEGY_READER_HPP
