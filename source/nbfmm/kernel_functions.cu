////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/kernel_functions.cu
/// @brief   The implementation of kernel functions
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/kernel_functions.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM
//
namespace nbfmm {

// The kernel function of gravity
__device__
float2 kernelGravity( const float2 position, const float weight ) {
  float2 effect;
  float r = sqrt(position.x * position.x + position.y * position.y);
  float tmp = weight / (r*r*r);
  effect.x = position.x * tmp;
  effect.y = position.y * tmp;
  return effect;
}

}  // namespace nbfmm
