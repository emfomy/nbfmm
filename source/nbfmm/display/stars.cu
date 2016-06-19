////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars.cu
/// @brief   The implementation of the class of stars
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/display/stars.hpp>

// The namespace NBFMM
namespace nbfmm {

// Default constructor
Stars::Stars(
    const int    fmm_level,
    const int    num_star,
    const int    width,
    const int    height,
    const int    fps,
    const float  tick,
    const float  grav_const,
    const float  size_scale,
    const float4 position_limits,
    const float4 display_limits
) : solver_(fmm_level, num_star, position_limits),
    num_star_(num_star),
    width_(width),
    height_(height),
    fps_(fps),
    tick_(tick),
    grav_const_(grav_const),
    size_scale_(size_scale),
    position_limits_(position_limits),
    display_limits_(display_limits)
{
  assert(num_star >= 0);
  assert(width > 0);
  assert(height > 0);
  assert(fps > 0);
  assert(tick > 0);
  assert(size_scale > 0);
  assert(position_limits.x < position_limits.z && position_limits.y < position_limits.w);
  assert(display_limits.x < display_limits.z && display_limits.y < display_limits.w);

  cudaMalloc(&gpuptr_position_cur_,  num_star * sizeof(float2));
  cudaMalloc(&gpuptr_position_pre_,  num_star * sizeof(float2));
  cudaMalloc(&gpuptr_acceleration_,  num_star * sizeof(float2));
  cudaMalloc(&gpuptr_weight_,        num_star * sizeof(float));
}

// Default destructor
Stars::~Stars() {
  cudaFree(gpuptr_position_cur_);
  cudaFree(gpuptr_position_pre_);
  cudaFree(gpuptr_acceleration_);
  cudaFree(gpuptr_weight_);
}

}  // namespace nbfmm
