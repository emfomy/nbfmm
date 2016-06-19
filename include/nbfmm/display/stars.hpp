////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/display/stars.hpp
/// @brief   The definition of the class of stars
///
/// @author  Mu Yang       <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang  <davidzan830@gmail.com>
///

/// @cond

#ifndef DEMO_STARS_DISPLAY_HPP_
#define DEMO_STARS_DISPLAY_HPP_

#include <nbfmm/config.hpp>
#include <cstdint>
#include <nbfmm/core.hpp>
#include <SyncedMemory.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM.
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The class of stars.
///
class Stars {

 protected:

  /// the FMM solver
  Solver solver_;

  /// the number of stars
  int num_star_;

  /// the width of the frame
  int width_;

  /// the height of the frame
  int height_;

  /// the number of frames per second
  const int FPS_;

  /// the gravitational constant
  const float grav_const_;

  /// the step size in time
  const float tick_;

  /// the scale of star size
  const float size_scale_;

  /// the position limits
  const float4 position_limits_;

  /// the display limits
  const float4 display_limits_;

  /// current position of stars
  float2* gpuptr_position_cur_;

  /// previous position of stars
  float2* gpuptr_position_pre_;

  /// the acceleration of stars
  float2* gpuptr_acceleration_;

  /// the weight of stars
  float* gpuptr_weight_;

 public:

  // Constructor
  Stars( const int fmm_level, const int num_star, const int width, const int height, const int FPS,
         const float grav_const, const float tick, const float size_scale,
         const float4 position_limits, const float4 display_limits );

  /// Destructor
  ~Stars();

  /// Initialize
  void initialize();

  /// Update
  void update();

  /// Visualize
  void display( uint8_t *board );

  void deletion_check();
};

}  // namespace nbfmm

#endif  // DEMO_STARS_DISPLAY_HPP_

/// @endcond
