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
    const float4         position_limits,
    const KernelFunction kernel_function
) : num_level_(num_level),
    base_size_(1 << (num_level-1)),
    max_num_particle_(max_num_particle),
    position_limits_(position_limits),
    kernel_function_(kernel_function) {
  cudaMalloc(&gpuptr_position, max_num_particle_ * sizeof(float2));
  cudaMalloc(&gpuptr_effect,   max_num_particle_ * sizeof(float2));
  cudaMalloc(&gpuptr_weight,   max_num_particle_ * sizeof(float));
  cudaMalloc(&gpuptr_index,    max_num_particle_ * sizeof(int));
  cudaMalloc(&gpuptr_perm,     max_num_particle_ * sizeof(int));
  cudaMalloc(&gpuptr_head,     (base_size_*base_size_+1) * sizeof(int));
  cudaMalloc3D(&pitchedptr_multipole, make_cudaExtent(base_size_*sizeof(float),  base_size_, num_level_));
  cudaMalloc3D(&pitchedptr_local,     make_cudaExtent(base_size_*sizeof(float2), base_size_, num_level_));
}

// Default destructor
Solver::~Solver() {
  cudaFree(gpuptr_position);
  cudaFree(gpuptr_effect);
  cudaFree(gpuptr_weight);
  cudaFree(gpuptr_index);
  cudaFree(gpuptr_perm);
  cudaFree(gpuptr_head);
  cudaFree(gpuptr_multipole);
  cudaFree(gpuptr_local);
}

}  // namespace nbfmm
