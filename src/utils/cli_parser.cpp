#include "utils/cli_parser.hpp"
#include <cstring>
#include <expected>
#include <format>
#include <print>
#include <string>

namespace starfwi {
namespace utils {

void print_usage(const char *program_name) {
  std::println(stderr, "Usage: {} [options] <segy_file_path> [timesteps]",
               program_name);
  std::println(stderr, "");
  std::println(stderr, "Options:");
  std::println(stderr, "  -v, --verbose              Enable verbose output");
  std::println(stderr, "  --snapshots N              Save wavefield snapshots "
                       "every N timesteps");
  std::println(stderr, "  --snapshot-dir DIR         Output directory for "
                       "snapshots (default: ./output)");
  std::println(stderr, "  --shots N                  Number of shots to "
                       "simulate (default: use SEG-Y data or 1)");
  std::println(stderr, "");
  std::println(stderr, "FWI Options:");
  std::println(stderr, "  --generate-observed        Generate observed data "
                       "from true model and save");
  std::println(stderr, "  --observed-dir DIR         Directory for observed "
                       "data (default: ./observed)");
  std::println(stderr, "");
  std::println(stderr, "Arguments:");
  std::println(stderr,
               "  segy_file_path   Path to SEG-Y velocity model file or "
               ".tar.gz archive");
  std::println(stderr, "  timesteps        (Optional) Number of time steps for "
                       "wave propagation");
  std::println(stderr, "");
  std::println(stderr, "Examples:");
  std::println(stderr, "  {} velocity_model.segy", program_name);
  std::println(stderr, "  {} -v velocity_model.segy 500", program_name);
  std::println(
      stderr,
      "  {} --snapshots 10 --snapshot-dir ./viz velocity_model.segy 100",
      program_name);
  std::println(stderr, "  {} --shots 8 --snapshots 50 velocity_model.segy",
               program_name);
  std::println(stderr, "");
  std::println(stderr, "FWI Workflow:");
  std::println(stderr, "  # Step 1: Generate observed data from true model");
  std::println(stderr,
               "  {} --generate-observed --observed-dir ./data true_model.segy",
               program_name);
  std::println(stderr, "  # Step 2: Run FWI with initial model (loads observed "
                       "from --observed-dir)");
  std::println(stderr, "  {} --observed-dir ./data initial_model.segy",
               program_name);
}

std::expected<CliArgs, std::string> parse_command_line(int argc, char **argv,
                                                       int rank) {
  CliArgs args;

  // Default directories
  args.snapshot_dir = "./output";
  args.observed_dir = "./observed";

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

    if (arg == "--shots") {
      if (i + 1 >= argc) {
        return std::unexpected("--shots requires an argument");
      }
      try {
        args.num_shots = std::stoi(argv[++i]);
        if (args.num_shots <= 0) {
          return std::unexpected(std::format(
              "--shots must be a positive integer (got: {})", args.num_shots));
        }
      } catch (const std::exception &e) {
        return std::unexpected(
            std::format("Invalid --shots value: '{}'", argv[i]));
      }
      continue;
    }

    if (arg == "--generate-observed") {
      args.generate_observed = true;
      continue;
    }

    if (arg == "--observed-dir") {
      if (i + 1 >= argc) {
        return std::unexpected("--observed-dir requires an argument");
      }
      args.observed_dir = argv[++i];
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
      // Second positional: timesteps
      try {
        args.num_timesteps = std::stoi(arg);
        if (args.num_timesteps <= 0) {
          return std::unexpected(
              std::format("timesteps must be a positive integer (got: {})",
                          args.num_timesteps));
        }
      } catch (const std::exception &e) {
        return std::unexpected(std::format(
            "Invalid timesteps parameter: '{}' ({})", arg, e.what()));
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
