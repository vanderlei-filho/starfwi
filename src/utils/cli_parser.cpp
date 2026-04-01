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
  std::println(stderr, "  --observed-dir DIR         Directory for observed "
                       "seismogram files (default: ./observed)");
  std::println(stderr, "  --velocity-scale F         Scale factor applied to "
                       "velocity model before inversion (default: 1.0)");
  std::println(stderr, "  --iterations N             Number of outer FWI "
                       "iterations (default: 10)");
  std::println(stderr, "  --step-length F            Max velocity update per "
                       "iteration in m/s (gradient is normalized, default: 50)");
  std::println(stderr, "");
  std::println(stderr, "Fault-Tolerance Options:");
  std::println(stderr, "  --checkpoint-dir DIR       Shared filesystem path "
                       "for persistent checkpoints");
  std::println(stderr, "  --checkpoint-interval N    Flush checkpoint every N "
                       "shots (default: 0 = disabled)");
  std::println(stderr, "  --wavefield-storage MODE   Forward wavefield storage "
                       "strategy: memory or disk (default: disk)");
  std::println(stderr, "  --wavefield-dir DIR        Directory for temporary "
                       "wavefield files when using disk strategy (default: /tmp)");
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
  std::println(stderr, "  Observed data is generated automatically from the "
                       "loaded velocity model before inversion begins.");
  std::println(stderr, "  A small perturbation is applied to the model to "
                       "create a non-trivial starting point for inversion.");
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

    if (arg == "--observed-dir") {
      if (i + 1 >= argc)
        return std::unexpected("--observed-dir requires an argument");
      args.observed_dir = argv[++i];
      continue;
    }

    if (arg == "--velocity-scale") {
      if (i + 1 >= argc)
        return std::unexpected("--velocity-scale requires an argument");
      try {
        args.velocity_scale = std::stof(argv[++i]);
      } catch (const std::exception &) {
        return std::unexpected(
            std::format("Invalid --velocity-scale value: '{}'", argv[i]));
      }
      continue;
    }

    if (arg == "--checkpoint-dir") {
      if (i + 1 >= argc) {
        return std::unexpected("--checkpoint-dir requires an argument");
      }
      args.checkpoint_dir = argv[++i];
      continue;
    }

    if (arg == "--wavefield-storage") {
      if (i + 1 >= argc)
        return std::unexpected("--wavefield-storage requires an argument");
      std::string mode = argv[++i];
      if (mode == "memory")
        args.wavefield_storage = CliArgs::WavefieldStorage::MEMORY;
      else if (mode == "disk")
        args.wavefield_storage = CliArgs::WavefieldStorage::DISK;
      else
        return std::unexpected(
            std::format("Invalid --wavefield-storage value: '{}' (expected: memory or disk)", mode));
      continue;
    }

    if (arg == "--wavefield-dir") {
      if (i + 1 >= argc)
        return std::unexpected("--wavefield-dir requires an argument");
      args.wavefield_dir = argv[++i];
      continue;
    }

    if (arg == "--iterations") {
      if (i + 1 >= argc)
        return std::unexpected("--iterations requires an argument");
      try {
        args.num_iterations = std::stoi(argv[++i]);
        if (args.num_iterations <= 0)
          return std::unexpected(std::format(
              "--iterations must be a positive integer (got: {})", args.num_iterations));
      } catch (const std::exception &) {
        return std::unexpected(
            std::format("Invalid --iterations value: '{}'", argv[i]));
      }
      continue;
    }

    if (arg == "--step-length") {
      if (i + 1 >= argc)
        return std::unexpected("--step-length requires an argument");
      try {
        args.step_length = std::stof(argv[++i]);
      } catch (const std::exception &) {
        return std::unexpected(
            std::format("Invalid --step-length value: '{}'", argv[i]));
      }
      continue;
    }

    if (arg == "--checkpoint-interval") {
      if (i + 1 >= argc) {
        return std::unexpected("--checkpoint-interval requires an argument");
      }
      try {
        int val = std::stoi(argv[++i]);
        if (val <= 0) {
          return std::unexpected(std::format(
              "--checkpoint-interval must be a positive integer (got: {})", val));
        }
        args.checkpoint_interval = static_cast<size_t>(val);
      } catch (const std::exception &e) {
        return std::unexpected(
            std::format("Invalid --checkpoint-interval value: '{}'", argv[i]));
      }
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
