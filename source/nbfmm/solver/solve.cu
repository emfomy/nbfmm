////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/solve.cu
/// @brief   Solve system
///
/// @author  Mu Yang <emfomy@gmail.com>
///
#include <math.h>
#include <cuda_runtime.h>
#include <nbfmm/solver.hpp>

partical_find_grid(int num_particle,float left_bound,float down_bound,float gridWidth,float gridHeight,float2* gpuptr_position,int2* gpuptr_index)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=num_particle)
		return;
	gpuptr_index[idx].x=floorf((gpuptr_position[idx].x-left_bound)/gridWidth);
	gpuptr_index[idx].y=floorf((gpuptr_position[idx].y-down_bound)/gridHeight);
}

//  The namespace NBFMM
namespace nbfmm {

// Solve system
void Solver::solve(
    const int num_particle,
    const float2* gpuptr_position_origin,
    const float* gpuptr_weight,
    const float2* gpuptr_effect_origin
) {
	float gridWidth=(position_limits_.z-position_limits_.x)/base_size_;
	float gridHeight=(position_limits_.w-position_limits_.y)/base_size_;
	int KERNEL_blockSize_pointwise=1024;
	int KERNEL_gridSize_pointwise=((num_particle-1)/KERNEL_blockSize_pointwise)+1;
	partical_find_grid<<<KERNEL_gridSize_pointwise,KERNEL_blockSize_pointwise>>>(num_particle,position_limits_.x,position_limits_.y,gridWidth,gridHeight, gpuptr_position,gpuptr_index);
  /// @todo Implement!
}

}  // namespace nbfmm
