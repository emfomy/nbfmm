////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/solve.cu
/// @brief   Solve system
///
/// @author  Mu Yang <emfomy@gmail.com>
///
#include <math.h>
#include <cuda_runtime.h>
#include <nbfmm/solver.hpp>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

__global__ void partical_find_grid(int num_particle,float left_bound,float down_bound,float gridWidth,float gridHeight,int base_size,const float2* gpuptr_position_origin,int* gpuptr_sortingIndex)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=num_particle)
		return;
	int posx=floorf((gpuptr_position_origin[idx].x-left_bound)/gridWidth);
	int posy=floorf((gpuptr_position_origin[idx].y-down_bound)/gridHeight);
	gpuptr_sortingIndex[idx]=posy*base_size+posx;
}
__global__ void sorting_input(int num_particle,int* gpuptr_perm_,const float2* gpuptr_position_origin,const float* gpuptr_weight_origin,float2* gpuptr_position_,float* gpuptr_weight_,int* gpuptr_sortingIndex,int2* gpuptr_index,int base_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=num_particle)
		return;
	gpuptr_position_[idx]=gpuptr_position_origin[gpuptr_perm_[idx]];
	gpuptr_weight_[idx]=gpuptr_weight_origin[gpuptr_perm_[idx]];
	gpuptr_index[idx].x=gpuptr_sortingIndex[idx]%base_size;
	gpuptr_index[idx].y=gpuptr_sortingIndex[idx]/base_size;

}

__global__ void extract_head(int num_particle,int* gpuptr_sortingIndex,int* gpuptr_head_)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=num_particle)
		return;
	if (idx==0)
		gpuptr_head_[gpuptr_sortingIndex[idx]]=idx;
	else
	{
		if (gpuptr_sortingIndex[idx]!=gpuptr_sortingIndex[idx-1])
			gpuptr_head_[gpuptr_sortingIndex[idx]]=idx;
	}
	gpuptr_head_[gpuptr_sortingIndex[num_particle]+1]=num_particle+1;
}
//  The namespace NBFMM
namespace nbfmm {

// Solve system
void Solver::solve(
    const int     num_particle,
    const float2* gpuptr_position_origin,
    const float*  gpuptr_weight_origin,
    const float2* gpuptr_effect_origin
) {
	float gridWidth=(position_limits_.z-position_limits_.x)/base_size_;
	float gridHeight=(position_limits_.w-position_limits_.y)/base_size_;
	int KERNEL_blockSize_pointwise=1024;
	int KERNEL_gridSize_pointwise=((num_particle-1)/KERNEL_blockSize_pointwise)+1;

	int* gpuptr_sortingIndex;
	cudaMalloc(&gpuptr_sortingIndex,num_particle*sizeof(int));
	partical_find_grid<<<KERNEL_gridSize_pointwise,KERNEL_blockSize_pointwise>>>(num_particle,position_limits_.x,position_limits_.y,gridWidth,gridHeight,base_size_, gpuptr_position_origin,gpuptr_sortingIndex);
  thrust::device_ptr<int> trst_permu(gpuptr_perm_),trst_sortingIndex(gpuptr_sortingIndex); //step 2 start
	thrust::sequence( trst_permu, trst_permu+num_particle);

	thrust::sort_by_key( trst_permu, trst_permu+num_particle, trst_sortingIndex);
	sorting_input<<<KERNEL_gridSize_pointwise,KERNEL_blockSize_pointwise>>>(num_particle,gpuptr_perm_,gpuptr_position_origin,gpuptr_weight_origin, gpuptr_position_,gpuptr_weight_,gpuptr_sortingIndex,gpuptr_index_,base_size_);
	extract_head<<<KERNEL_gridSize_pointwise,KERNEL_blockSize_pointwise>>>(num_particle,gpuptr_sortingIndex,gpuptr_head_); //step3: fill array of starting index
  /// @todo Implement!
}

}  // namespace nbfmm
