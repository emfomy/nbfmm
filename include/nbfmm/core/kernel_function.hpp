////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/core/kernel_function.hpp
/// @brief   The definition of kernel functions
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_CORE_KERNEL_FUNCTIONS_HPP_
#define NBFMM_CORE_KERNEL_FUNCTIONS_HPP_

#include <nbfmm/config.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The kernel function of gravitation
///
/// @param  position_target  the position of target particle
/// @param  position_source  the position of source particle
/// @param  weight_source    the mass of source particle
///
/// @return                  acceleration of target particle
///

// The kernel function of gravitation
__host__ __device__ inline
float2 kernelFunction(
    const float2 position_target,
    const float2 position_source,
    const float  weight_source
) {
  const float epsilon = 1;
  float2 distance = make_float2(position_source.x - position_target.x, position_source.y - position_target.y);
  float r = sqrt(distance.x * distance.x + distance.y * distance.y);
  float tmp = weight_source / (r*r*r + epsilon);
  return make_float2(distance.x * tmp, distance.y * tmp);
}

}  // namespace nbfmm

#endif  // NBFMM_CORE_KERNEL_FUNCTIONS_HPP_
