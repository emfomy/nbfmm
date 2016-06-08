////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver.cu
/// @brief   The implementation of the FMM solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

//  The namespace NBFMM
namespace nbfmm {

// Default constructor
Solver::Solver(
    const int            num_level,
    const int            max_num_particle,
    const float4         position_limits
) : num_level_(num_level),
    base_size_(1 << (num_level-1)),
    max_num_particle_(max_num_particle),
    position_limits_(position_limits) {
  assert(num_level > 0);
  assert(max_num_particle > 0);
  assert(position_limits.x < position_limits.z);
  assert(position_limits.y < position_limits.w);

  cudaMalloc(&gpuptr_position_,      max_num_particle_ * sizeof(float2));
  cudaMalloc(&gpuptr_effect_,        max_num_particle_ * sizeof(float2));
  cudaMalloc(&gpuptr_weight_,        max_num_particle_ * sizeof(float));
  cudaMalloc(&gpuptr_index_,         max_num_particle_ * sizeof(int2));
  cudaMalloc(&gpuptr_perm_,          max_num_particle_ * sizeof(int));
  cudaMalloc(&gpuptr_head_,          (base_size_ * base_size_ + 1) * sizeof(int));
  cudaMalloc(&gpuptr_cell_position_, base_size_ * base_size_ * num_level_ * sizeof(float2));
  cudaMalloc(&gpuptr_cell_effect_,   base_size_ * base_size_ * num_level_ * sizeof(float2));
  cudaMalloc(&gpuptr_cell_weight_,   base_size_ * base_size_ * num_level_ * sizeof(float));
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
}

}  // namespace nbfmm
