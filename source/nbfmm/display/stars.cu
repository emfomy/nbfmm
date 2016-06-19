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
  assert(num_star_ >= 0);
  assert(width_ > 0);
  assert(height_ > 0);
  assert(fps_ > 0);
  assert(tick_ > 0);
  assert(grav_const_ > 0);
  assert(size_scale_ > 0);
  assert(position_limits_.x < position_limits_.z && position_limits_.y < position_limits_.w);
  assert(display_limits_.x  < display_limits_.z  && display_limits_.y  < display_limits_.w);

  cudaMalloc(&gpuptr_position_cur_,  num_star_ * sizeof(float2));
  cudaMalloc(&gpuptr_position_pre_,  num_star_ * sizeof(float2));
  cudaMalloc(&gpuptr_acceleration_,  num_star_ * sizeof(float2));
  cudaMalloc(&gpuptr_weight_,        num_star_ * sizeof(float));
  cudaMalloc(&gpuptr_elimination_,   num_star_ * sizeof(bool));
}

// Default destructor
Stars::~Stars() {
  cudaFree(gpuptr_position_cur_);
  cudaFree(gpuptr_position_pre_);
  cudaFree(gpuptr_acceleration_);
  cudaFree(gpuptr_weight_);
  cudaFree(gpuptr_elimination_);
}

}  // namespace nbfmm
