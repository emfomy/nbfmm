////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/kernel_functions.hpp
/// @brief   The definition of kernel functions
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_KERNEL_FUNCTIONS_HPP_
#define NBFMM_KERNEL_FUNCTIONS_HPP_

#include <nbfmm/config.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The type alias of kernel function
///
typedef float2 (*KernelFunction)(const float2, const float);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The acceleration of gravity
///
/// @param  position  the position of particle
/// @param  weight    the mass of particle
///
/// @return           the acceleration of particle
///
inline float2 KernelGravity( const float2 position, const float weight ) {
  float2 effect;
  float r = sqrt(position.x * position.x + position.y * position.y);
  float tmp = weight / (r*r*r);
  effect.x = position.x * tmp;
  effect.y = position.y * tmp;
  return effect;
}

}  // namespace nbfmm

#endif  // NBFMM_KERNEL_FUNCTIONS_HPP_
