#include "utils/cli_parser.hpp"
#include <cstring>
#include <expected>
#include <format>
#include <print>
#include <string>

namespace starfwi {
namespace utils {

void print_usage(const char *program_name) {
  std::println(stderr, "Usage: {} [options] <segy_file_path> [iterations]",
               program_name);
  std::println(stderr, "");
  std::println(stderr, "Options:");
  std::println(stderr, "  -v, --verbose              Enable verbose output");
  std::println(stderr, "  --snapshots N              Save wavefield snapshots every N timesteps");
  std::println(stderr, "  --snapshot-dir DIR         Output directory for snapshots (default: ./output)");
  std::println(stderr, "");
  std::println(stderr, "Arguments:");
  std::println(stderr,
               "  segy_file_path   Path to SEG-Y velocity model file or "
               ".tar.gz archive");
  std::println(
      stderr,
      "  iterations       (Optional) Number of time steps to run (overrides "
      "default/config)");
  std::println(stderr, "");
  std::println(stderr, "Examples:");
  std::println(stderr, "  {} velocity_model.segy", program_name);
  std::println(stderr, "  {} -v velocity_model.segy 500", program_name);
  std::println(stderr, "  {} --snapshots 10 --snapshot-dir ./viz velocity_model.segy 100", program_name);
}

std::expected<CliArgs, std::string> parse_command_line(int argc, char **argv,
                                                       int rank) {
  CliArgs args;

  // Default snapshot directory
  args.snapshot_dir = "./output";

  // Parse arguments
  int positional_count = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    // Check for flags
    if (arg == "-v" || arg == "--verbose") {
      args.verbose = true;
      continue;
    }

    if (arg == "--snapshots") {
      if (i + 1 >= argc) {
        return std::unexpected("--snapshots requires an argument");
      }
      try {
        args.snapshot_interval = std::stoul(argv[++i]);
      } catch (const std::exception &e) {
        return std::unexpected(
            std::format("Invalid snapshot interval: '{}'", argv[i]));
      }
      continue;
    }

    if (arg == "--snapshot-dir") {
      if (i + 1 >= argc) {
        return std::unexpected("--snapshot-dir requires an argument");
      }
      args.snapshot_dir = argv[++i];
      continue;
    }

    // Check for unknown flags
    if (arg[0] == '-') {
      return std::unexpected(std::format("Unknown option '{}'", arg));
    }

    // Handle positional arguments
    if (positional_count == 0) {
      // First positional: segy_file_path
      args.segy_filepath = arg;
      positional_count++;
    } else if (positional_count == 1) {
      // Second positional: iterations
      try {
        args.num_iterations = std::stoi(arg);
        if (args.num_iterations <= 0) {
          return std::unexpected(
              std::format("iterations must be a positive integer (got: {})",
                          args.num_iterations));
        }
      } catch (const std::exception &e) {
        return std::unexpected(std::format(
            "Invalid iterations parameter: '{}' ({})", arg, e.what()));
      }
      positional_count++;
    } else {
      // Too many positional arguments
      return std::unexpected("Too many arguments");
    }
  }

  // Check that required argument was provided
  if (args.segy_filepath.empty()) {
    return std::unexpected("Missing required argument <segy_file_path>");
  }

  return args;
}

void print_ascii_art_title() {
  std::println("\n");
  std::println("  ______    __                          ________  __       "
               "__  ______ ");
  std::println(" /      \\  /  |                        /        |/  |  _  /  "
               "|/      |");
  std::println("/$$$$$$  |_$$ |_     ______    ______  $$$$$$$$/ $$ | / \\ "
               "$$ |$$$$$$/");
  std::println(
      "$$ \\__$$// $$   |   /      \\  /      \\ $$ |__    $$ |/$  \\$$ "
      "|  $$ |  ");
  std::println(
      "$$      \\$$$$$$/    $$$$$$  |/$$$$$$  |$$    |   $$ /$$$  $$ |  "
      "$$ |  ");
  std::println(" $$$$$$  | $$ | __  /    $$ |$$ |  $$/ $$$$$/    $$ $$/$$ $$ "
               "|  $$ |  ");
  std::println(
      "/  \\__$$ | $$ |/  |/$$$$$$$ |$$ |      $$ |      $$$$/  $$$$ | "
      "_$$ |_ ");
  std::println("$$    $$/  $$  $$/ $$    $$ |$$ |      $$ |      $$$/    $$$ "
               "|/ $$   |");
  std::println(" $$$$$$/    $$$$/   $$$$$$$/ $$/       $$/       $$/      "
               "$$/ $$$$$$/ ");
  std::println("\n");
}

} // namespace utils
} // namespace starfwi
