////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver.cu
/// @brief   The implementation of the FMM solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core/solver.hpp>

// The NBFMM namespace
namespace nbfmm {

// Default constructor
Solver::Solver(
    const int            num_level,
    const int            max_num_particle,
    const float4         position_limits
) : num_level_(num_level),
    base_dim_(1 << (num_level+1)),
    num_cell_p1_(base_dim_*base_dim_+1),
    max_num_particle_(max_num_particle),
    position_limits_(position_limits)
{
  assert(num_level >= 0);
  assert(base_dim_ < kMaxBlockDim);
  assert(max_num_particle > 0 && max_num_particle < kMaxGridDim);
  assert(position_limits.x < position_limits.z && position_limits.y < position_limits.w);

  cudaMalloc(&gpuptr_position_,      max_num_particle_ * sizeof(float2));
  cudaMalloc(&gpuptr_effect_,        max_num_particle_ * sizeof(float2));
  cudaMalloc(&gpuptr_weight_,        max_num_particle_ * sizeof(float));
  cudaMalloc(&gpuptr_index_,         max_num_particle_ * sizeof(int2));
  cudaMalloc(&gpuptr_perm_,          max_num_particle_ * sizeof(int));
  cudaMalloc(&gpuptr_head_,          num_cell_p1_      * sizeof(int));
#pragma warning
  if ( num_level_ < 2 ) {
    cudaMalloc(&gpuptr_cell_position_, base_dim_ * base_dim_ * 2 * sizeof(float2));
    cudaMalloc(&gpuptr_cell_effect_,   base_dim_ * base_dim_ * 2 * sizeof(float2));
    cudaMalloc(&gpuptr_cell_weight_,   base_dim_ * base_dim_ * 2 * sizeof(float));
  } else {
    cudaMalloc(&gpuptr_cell_position_, base_dim_ * base_dim_ * num_level_ * sizeof(float2));
    cudaMalloc(&gpuptr_cell_effect_,   base_dim_ * base_dim_ * num_level_ * sizeof(float2));
    cudaMalloc(&gpuptr_cell_weight_,   base_dim_ * base_dim_ * num_level_ * sizeof(float));
  }
  cudaMalloc(&gpuptr_buffer_float2_, max_num_particle_     * sizeof(float2));
  cudaMalloc(&gpuptr_buffer_int2_,   base_dim_ * base_dim_ * sizeof(int2));
}

// Default destructor
Solver::~Solver() {
  cudaFree(gpuptr_position_);
  cudaFree(gpuptr_effect_);
  cudaFree(gpuptr_weight_);
  cudaFree(gpuptr_index_);
  cudaFree(gpuptr_perm_);
  cudaFree(gpuptr_head_);
  cudaFree(gpuptr_cell_position_);
  cudaFree(gpuptr_cell_effect_);
  cudaFree(gpuptr_cell_weight_);
  cudaFree(gpuptr_buffer_float2_);
  cudaFree(gpuptr_buffer_int2_);
}

}  // namespace nbfmm
