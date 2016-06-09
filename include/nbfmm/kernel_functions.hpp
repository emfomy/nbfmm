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
/// The kernel function of gravity
///
/// @param  position  the position of particle
/// @param  weight    the mass of particle
///
/// @return           the acceleration of particle
///

// The kernel function of gravity
__host__ __device__ inline
float2 kernelFunction( const float2 position, const float weight ) {
  float r = sqrt(position.x * position.x + position.y * position.y);
  float tmp = weight / (r*r*r);
  return make_float2(position.x * tmp, position.y * tmp);
}

}  // namespace nbfmm

#endif  // NBFMM_KERNEL_FUNCTIONS_HPP_
