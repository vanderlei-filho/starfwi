#include "utils/segy_loader.hpp"
#include <cmath>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <iostream>
#include <print>

namespace starfwi {

static std::string format_number(size_t number) {
  std::string num_str = std::to_string(number);
  std::string formatted;
  int count = 0;

  for (int i = num_str.length() - 1; i >= 0; --i) {
    if (count > 0 && count % 3 == 0) {
      formatted = ',' + formatted;
    }
    formatted = num_str[i] + formatted;
    count++;
  }

  return formatted;
}

static size_t calculate_model_memory(const SimulationConfig &config) {
  size_t total_memory = 0;

  // Velocity model data (primary memory consumer)
  if (config.velocity_model.is_loaded) {
    total_memory += config.velocity_model.data.size() * sizeof(float);
  }

  // Source positions (minimal - only a few shots)
  total_memory += config.source.x_positions.size() * sizeof(float);
  total_memory += config.source.y_positions.size() * sizeof(float);
  total_memory += config.source.z_positions.size() * sizeof(float);

  return total_memory;
}

// SEG-Y Binary File Header structure (400 bytes)
struct SegyBinaryHeader {
  int32_t job_id;
  int32_t line_number;
  int32_t reel_number;
  int16_t traces_per_ensemble;
  int16_t aux_traces_per_ensemble;
  int16_t sample_interval; // Microseconds
  int16_t original_sample_interval;
  int16_t samples_per_trace;
  int16_t original_samples_per_trace;
  int16_t data_sample_format; // 1=IBM float, 5=IEEE float
  int16_t ensemble_fold;
  int16_t trace_sorting;
  int16_t vertical_sum;
  int16_t sweep_freq_start;
  int16_t sweep_freq_end;
  int16_t sweep_length;
  int16_t sweep_type;
  int16_t sweep_trace_number;
  int16_t sweep_taper_length_start;
  int16_t sweep_taper_length_end;
  int16_t taper_type;
  int16_t correlated;
  int16_t binary_gain_recovered;
  int16_t amplitude_recovery;
  int16_t measurement_system; // 1=meters, 2=feet
  int16_t impulse_polarity;
  int16_t vibratory_polarity;
  char unassigned[240];
  int16_t segy_revision;
  int16_t fixed_length_trace;
  int16_t extended_headers;
  char unassigned2[94];
};

// SEG-Y Trace Header structure (240 bytes)
// We only need a subset of fields for coordinate extraction
struct SegyTraceHeader {
  int32_t trace_sequence_line;      // Bytes 1-4
  int32_t trace_sequence_file;      // Bytes 5-8
  int32_t field_record;             // Bytes 9-12
  int32_t trace_number;             // Bytes 13-16
  int32_t source_point;             // Bytes 17-20
  int32_t ensemble_number;          // Bytes 21-24 (CDP number)
  int32_t trace_in_ensemble;        // Bytes 25-28
  int16_t trace_id;                 // Bytes 29-30
  int16_t vertical_sum;             // Bytes 31-32
  int16_t horizontal_stack;         // Bytes 33-34
  int16_t data_use;                 // Bytes 35-36
  int32_t source_receiver_distance; // Bytes 37-40
  int32_t receiver_elevation;       // Bytes 41-44
  int32_t source_elevation;         // Bytes 45-48
  int32_t source_depth;             // Bytes 49-52
  int32_t receiver_datum_elevation; // Bytes 53-56
  int32_t source_datum_elevation;   // Bytes 57-60
  int32_t source_water_depth;       // Bytes 61-64
  int32_t receiver_water_depth;     // Bytes 65-68
  int16_t elevation_scalar;         // Bytes 69-70
  int16_t coordinate_scalar;        // Bytes 71-72
  int32_t source_x;                 // Bytes 73-76
  int32_t source_y;                 // Bytes 77-80
  int32_t receiver_x;               // Bytes 81-84
  int32_t receiver_y;               // Bytes 85-88
  char remaining[152];              // Bytes 89-240
};

// Helper function to swap bytes for endianness
template <typename T> T swap_endian(T value) {
  union {
    T value;
    unsigned char bytes[sizeof(T)];
  } src, dst;

  src.value = value;
  for (size_t i = 0; i < sizeof(T); i++) {
    dst.bytes[i] = src.bytes[sizeof(T) - 1 - i];
  }
  return dst.value;
}

// Convert IBM float to IEEE float
float ibm_to_ieee(int32_t ibm) {
  if (ibm == 0)
    return 0.0f;

  int32_t sign = ibm >> 31;
  int32_t exponent = ((ibm >> 24) & 0x7F) - 64;
  int32_t mantissa = ibm & 0x00FFFFFF;

  float value = static_cast<float>(mantissa) / 16777216.0f;
  value = std::ldexp(value, exponent * 4);

  return sign ? -value : value;
}

std::expected<void, std::string>
SEGYLoader::load_segy_model(const std::string &filepath, bool verbose) {
  std::ifstream file(filepath, std::ios::binary);

  if (verbose) {
    std::println("[starfwi][segy_loader] Loading SEG-Y model from file {}...",
                 filepath);
  }

  if (!file.is_open()) {
    return std::unexpected(std::format("Cannot open SEG-Y file: {}", filepath));
  }

  // Read textual header (3200 bytes) - 40 lines of 80 characters
  char textual_header[3200];
  file.read(textual_header, 3200);

  // Read binary header (400 bytes)
  SegyBinaryHeader bin_header;
  file.read(reinterpret_cast<char *>(&bin_header), 400);

  // Swap endianness if needed (SEG-Y is big-endian)
  int32_t job_id = swap_endian(bin_header.job_id);
  int32_t line_number = swap_endian(bin_header.line_number);
  int32_t reel_number = swap_endian(bin_header.reel_number);
  int16_t traces_per_ensemble = swap_endian(bin_header.traces_per_ensemble);
  int16_t aux_traces_per_ensemble =
      swap_endian(bin_header.aux_traces_per_ensemble);
  int16_t sample_interval = swap_endian(bin_header.sample_interval);
  int16_t original_sample_interval =
      swap_endian(bin_header.original_sample_interval);
  int16_t samples_per_trace = swap_endian(bin_header.samples_per_trace);
  int16_t original_samples_per_trace =
      swap_endian(bin_header.original_samples_per_trace);
  int16_t data_format = swap_endian(bin_header.data_sample_format);
  int16_t ensemble_fold = swap_endian(bin_header.ensemble_fold);
  int16_t trace_sorting = swap_endian(bin_header.trace_sorting);
  int16_t measurement_system = swap_endian(bin_header.measurement_system);
  int16_t segy_revision = swap_endian(bin_header.segy_revision);
  int16_t fixed_length_trace = swap_endian(bin_header.fixed_length_trace);
  int16_t extended_headers = swap_endian(bin_header.extended_headers);

  // Read traces and extract coordinates
  size_t trace_count = 0;
  size_t bytes_per_sample = 4; // 4 bytes for float
  size_t trace_data_size = samples_per_trace * bytes_per_sample;

  // Store coordinates from trace headers
  std::vector<int32_t> x_coords;
  std::vector<int32_t> y_coords;
  int16_t coord_scalar = 1; // Scalar to apply to coordinates

  while (file.peek() != EOF) {
    // Read trace header (240 bytes)
    SegyTraceHeader trace_header;
    file.read(reinterpret_cast<char *>(&trace_header), 240);

    if (file.gcount() != 240)
      break;

    // Extract coordinates from first trace to get scalar
    if (trace_count == 0) {
      coord_scalar = swap_endian(trace_header.coordinate_scalar);
    }

    // Extract X/Y coordinates (stored at bytes 73-76 and 77-80)
    int32_t x_coord = swap_endian(trace_header.source_x);
    int32_t y_coord = swap_endian(trace_header.source_y);
    x_coords.push_back(x_coord);
    y_coords.push_back(y_coord);

    // Read trace data
    std::vector<char> trace_data(trace_data_size);
    file.read(trace_data.data(), trace_data_size);

    if (file.gcount() != static_cast<std::streamsize>(trace_data_size))
      break;

    // Convert data based on format
    for (size_t i = 0; i < samples_per_trace; ++i) {
      int32_t raw_value;
      std::memcpy(&raw_value, &trace_data[i * 4], 4);
      raw_value = swap_endian(raw_value);

      float value;
      if (data_format == 1) {
        value = ibm_to_ieee(raw_value);
      } else if (data_format == 5) {
        std::memcpy(&value, &raw_value, 4);
      } else {
        return std::unexpected(
            std::format("Unsupported data format: {}", data_format));
      }

      config_.velocity_model.data.push_back(value);
    }

    trace_count++;
  }

  file.close();

  // ========================================================================
  // GRID SPACING CALCULATION
  // ========================================================================
  // SEG-Y stores horizontal (X, Y) and vertical (Z) spacing differently:
  //
  // 1. X/Y SPACING (dx, dy):
  //    - Each trace header contains X and Y coordinates (bytes 73-76, 77-80)
  //    - We calculate spacing by taking the difference between consecutive
  //    traces
  //    - Example: If trace 1 is at X=0 and trace 2 is at X=1250, then dx=1250
  //
  // 2. Z SPACING (dz):
  //    - NOT stored in trace headers (no Z coordinate per sample)
  //    - Instead, Z is implicit: z = sample_index * dz
  //    - The dz value comes from the binary header's 'sample_interval' field
  //    - For velocity models, sample_interval (in μs) represents depth spacing
  //    - Example: sample_interval=1250 μs → dz = 1.25 meters (1250/1000)
  //
  // Note: For a trace with 2801 samples and dz=1.25m:
  //   - Sample 0: depth = 0 * 1.25 = 0.0 m
  //   - Sample 1: depth = 1 * 1.25 = 1.25 m
  //   - Sample 2800: depth = 2800 * 1.25 = 3500 m
  // ========================================================================

  // --- Calculate horizontal spacing (dx, dy) from trace header coordinates ---
  float dx_calculated = 0.0f;
  float dy_calculated = 0.0f;

  if (x_coords.size() >= 2) {
    // Apply coordinate scalar to convert stored coordinates to meters
    // SEG-Y standard: coordinate_scalar defines how to scale coordinates
    // If scalar > 0: multiply coordinates by scalar
    // If scalar < 0: divide coordinates by -scalar
    // If scalar == 0: coordinates are already in correct units
    float scale_factor = 1.0f;
    if (coord_scalar > 0) {
      scale_factor = coord_scalar;
    } else if (coord_scalar < 0) {
      scale_factor = 1.0f / (-coord_scalar);
    }

    // Convert trace coordinates from stored values to real-world meters
    float x0_meters = x_coords[0] * scale_factor;
    float x1_meters = x_coords[1] * scale_factor;
    float y0_meters = y_coords[0] * scale_factor;

    // Calculate dx from spacing between first two traces
    dx_calculated = std::abs(x1_meters - x0_meters);

    // Apply Marmousi2-specific correction: coordinates are multiplied by 1000
    // (This handles non-standard files where coord_scalar=0 but coords are
    // scaled)
    dx_calculated /= 1000.0f;
    x0_meters /= 1000.0f;
    x1_meters /= 1000.0f;

    // Calculate dy by searching for first Y-coordinate change
    // (For 2D models, all traces have the same Y, so dy will remain 0)
    for (size_t i = 1; i < y_coords.size(); ++i) {
      if (y_coords[i] != y_coords[0]) {
        float yi_meters = y_coords[i] * scale_factor;
        float yi_prev_meters = y_coords[i - 1] * scale_factor;
        dy_calculated = std::abs(yi_meters - yi_prev_meters);

        // Apply same Marmousi2 correction to dy
        dy_calculated /= 1000.0f;
        break;
      }
    }
  }

  // --- Calculate vertical spacing (dz) from binary header ---
  // For velocity models, the sample_interval field (normally in microseconds
  // for time) is repurposed to store depth spacing in a scaled format Example:
  // sample_interval=1250 μs represents dz=1.25 meters (1250/1000)
  float dz_from_interval = sample_interval / 1000.0f;

  // Infer grid dimensions from SEG-Y file
  if (config_.grid.nz == 0) {
    config_.grid.nz = samples_per_trace;
    config_.grid.dz = dz_from_interval;
  }

  if (config_.grid.nx == 0 && trace_count > 0) {
    config_.grid.nx = trace_count;

    // Detect if this is a 2D or 3D model by checking Y coordinate variation
    bool has_y_variation = false;
    if (y_coords.size() > 1) {
      for (size_t i = 1; i < y_coords.size(); ++i) {
        if (y_coords[i] != y_coords[0]) {
          has_y_variation = true;
          break;
        }
      }
    }

    // TODO: Add 3D model support
    config_.grid.ny = 1; // We assume a 2D model for now.
    if (has_y_variation) {
      return std::unexpected("Detected Y-coordinate variation (3D model), "
                             "but 3D support is not fully implemented.");
    }

    // Use calculated dx if available, otherwise fallback to dz
    if (dx_calculated > 0.0f) {
      config_.grid.dx = dx_calculated;
    } else {
      config_.grid.dx = dz_from_interval;
      if (verbose) {
        std::println("[starfwi][segy_loader] WARNING: Could not calculate dx "
                     "from coordinates, falling back to dz value...");
      }
    }

    // For 2D models (ny=1), set dy=0 since there's no spacing in Y direction
    if (dy_calculated > 0.0f) {
      config_.grid.dy = dy_calculated;
    } else {
      config_.grid.dy = 0.0f; // 2D model: no Y dimension spacing
    }
  }

  config_.velocity_model.is_loaded = true;

  return {};
}

} // namespace starfwi
