#ifndef CLI_PARSER_HPP
#define CLI_PARSER_HPP

#include <expected>
#include <string>

namespace starfwi {
namespace utils {

/**
 * @brief Structure to hold parsed command-line arguments
 */
struct CliArgs {
  std::string segy_filepath; ///< Path to SEG-Y file
  int num_iterations = -1;   ///< Number of iterations (-1 = use default)
  bool verbose = false;      ///< Enable verbose output
  size_t snapshot_interval =
      0; ///< Save snapshots every N timesteps (0 = disabled)
  std::string snapshot_dir = ""; ///< Output directory for snapshots
  int num_shots = -1; ///< Number of shots (-1 = use default or SEG-Y data)

  // FWI-related options
  bool generate_observed = false;    ///< Generate observed data from true model
  std::string observed_dir = "";     ///< Directory for observed data
};

/**
 * @brief Parse command-line arguments for starfwi
 *
 * Parses command-line arguments and validates them.
 *
 * Expected usage: starfwi [options] <segy_file_path> [iterations]
 *
 * Options:
 *   -v, --verbose    Enable verbose output (only on rank 0)
 *
 * @param argc Argument count from main()
 * @param argv Argument vector from main()
 * @param rank MPI rank (used to determine which rank should enable verbose
 * output)
 * @return Expected containing CliArgs on success, or error message string on
 * failure
 */
std::expected<CliArgs, std::string> parse_command_line(int argc, char **argv,
                                                       int rank);

/**
 * @brief Print usage information
 *
 * @param program_name The name of the program (typically argv[0])
 */
void print_usage(const char *program_name);

/**
 * @brief Print ASCII art title banner for StarFWI
 */
void print_ascii_art_title();

} // namespace utils
} // namespace starfwi

#endif // CLI_PARSER_HPP
