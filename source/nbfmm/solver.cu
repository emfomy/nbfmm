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
    const int num_level,
    const int max_num_point,
    const KernelFunction kernel_function
) : num_level_(num_level),
    size_base_grid_(1 << (num_level-1)),
    max_num_point_(max_num_point),
    kernel_function_(kernel_function) {
  cudaMalloc(&gpuptr_position, max_num_point_ * sizeof(float2));
  cudaMalloc(&gpuptr_effect,   max_num_point_ * sizeof(float2));
  cudaMalloc(&gpuptr_index,    max_num_point_ * sizeof(int));
  cudaMalloc(&gpuptr_head,     size_base_grid_ * size_base_grid_ * sizeof(int));
  cudaMalloc3D(&pitchedptr_multipole, make_cudaExtent(size_base_grid_*sizeof(float),  size_base_grid_, num_level_));
  cudaMalloc3D(&pitchedptr_local,     make_cudaExtent(size_base_grid_*sizeof(float2), size_base_grid_, num_level_));
  gpuptr_multipole = (float*)  pitchedptr_multipole.ptr;
  gpuptr_local     = (float2*) pitchedptr_local.ptr;
}

// Default destructor
Solver::~Solver() {
  cudaFree(gpuptr_position);
  cudaFree(gpuptr_effect);
  cudaFree(gpuptr_index);
  cudaFree(gpuptr_head);
  cudaFree(gpuptr_multipole);
  cudaFree(gpuptr_local);
}

}  // namespace nbfmm
