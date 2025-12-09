#ifndef RECEIVER_GEOMETRY_HPP
#define RECEIVER_GEOMETRY_HPP

#include <vector>

namespace starfwi {

/**
 * @brief Global receiver array geometry for seismic survey
 *
 * In a typical seismic survey, receivers are placed at fixed positions
 * (usually at the surface) and remain stationary while sources are moved
 * to different locations for each shot. All shots record data at the
 * same receiver positions.
 *
 * This structure holds the receiver positions that are shared across
 * all shots in the survey.
 */
struct ReceiverGeometry {
  std::vector<float> x; ///< X coordinates of receivers (meters)
  std::vector<float> y; ///< Y coordinates of receivers (meters)
  std::vector<float> z; ///< Z coordinates of receivers (meters)

  /**
   * @brief Get number of receivers
   */
  size_t size() const { return x.size(); }

  /**
   * @brief Check if receiver geometry is empty
   */
  bool empty() const { return x.empty(); }

  /**
   * @brief Clear all receiver positions
   */
  void clear() {
    x.clear();
    y.clear();
    z.clear();
  }

  /**
   * @brief Reserve space for n receivers
   */
  void reserve(size_t n) {
    x.reserve(n);
    y.reserve(n);
    z.reserve(n);
  }

  /**
   * @brief Add a receiver at position (rx, ry, rz)
   */
  void add_receiver(float rx, float ry, float rz) {
    x.push_back(rx);
    y.push_back(ry);
    z.push_back(rz);
  }
};

} // namespace starfwi

#endif // RECEIVER_GEOMETRY_HPP
