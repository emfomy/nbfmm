////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/display/stars.hpp
/// @brief   The definition of the class of stars
///
/// @author  Mu Yang       <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang  <davidzan830@gmail.com>
///

#ifndef NBFMM_DISPLAY_STARS_HPP_
#define NBFMM_DISPLAY_STARS_HPP_

#include <nbfmm/config.hpp>
#include <cstdint>
#include <nbfmm/core.hpp>
#include <SyncedMemory.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The NBFMM namespace.
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The class of stars.
///
class Stars {

 protected:

  /// The FMM solver
  Solver solver_;

  /// The number of stars
  int num_star_;

  /// The frame width
  int width_;

  /// The frame height
  int height_;

  /// The number of frames per second
  const int fps_;

  /// The step size in time
  const float tick_;

  /// The gravitational constant
  const float grav_const_;

  /// The scale of star size
  const float size_scale_;

  /// The limits of positions. [x_min, y_min, x_max, y_max].
  const float4 position_limits_;

  /// The limits of display positions. [x_min, y_min, x_max, y_max].
  const float4 display_limits_;

 public:

  /// The current positions of stars
  float2* gpuptr_position_cur_;

  /// The previous positions of stars
  float2* gpuptr_position_pre_;

  /// The acceleration of stars
  float2* gpuptr_acceleration_;

  /// The weight of stars
  float* gpuptr_weight_;

 public:

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Default constructor
  ///
  /// @param  fmm_level         the number of FMM cell levels.
  /// @param  num_star          the number of stars.
  /// @param  width             the frame width.
  /// @param  height            the frame height.
  /// @param  fps               the number of frames per second
  /// @param  tick              the step size in time
  /// @param  grav_const        the gravitational constant.
  /// @param  size_scale        the scale of star size
  /// @param  position_limits   the limits of positions. [x_min, y_min, x_max, y_max].
  /// @param  display_limits    the limits of display positions. [x_min, y_min, x_max, y_max].
  ///
  Stars( const int fmm_level, const int num_star, const int width, const int height, const int fps,
         const float tick, const float grav_const, const float size_scale,
         const float4 position_limits, const float4 display_limits );

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Default destructor
  ///
  ~Stars();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Initialize the data
  ///
  /// @tparam  Func  the function type
  /// @tparam  Args  the argument types
  ///
  /// @param   func  the initialization function
  /// @param   args  the arguments of the initialization function
  ///
  /// @note
  ///   the last arguments of @p func are #gpuptr_position_cur_, #gpuptr_position_pre_, #gpuptr_weight_.
  ///
  template <typename Func, typename... Args>
  void initialize(Func func, Args... args) {
    func(args..., gpuptr_position_cur_, gpuptr_position_pre_, gpuptr_weight_);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Update date
  ///
  void update();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Update date
  ///
  /// @param  board  the pixel board
  ///
  void display( uint8_t* board );

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Update date
  ///
  void checkDeletion();
};

}  // namespace nbfmm

#endif  // NBFMM_DISPLAY_STARS_HPP_
