////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars.cu
/// @brief   The implementation of stars class
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang <davidzan830@gmail.com>
///

/// @cond

#include <nbfmm/display.hpp>
#include <cstdint>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <nbfmm/utility.hpp>

__global__ void checkDeletion_kernel(int num_star, float2* gpuptr_position_cur, float4 position_limits, int* elimination)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=num_star )
    {
     return;
    }
    if (gpuptr_position_cur[idx].x<position_limits.x || gpuptr_position_cur[idx].y<position_limits.y ||gpuptr_position_cur[idx].x>position_limits.z ||gpuptr_position_cur[idx].y>position_limits.w)
    {
      elimination[idx] = 1;
    } else {
      elimination[idx] = 0;
    }
}

void nbfmm::Stars::checkDeletion()
{
  auto position_limits = position_limits_;
  int* elimination;
  cudaMalloc(&elimination, sizeof(int)*num_star_);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((num_star_-1)/kNumThread_pointwise)+1;
  checkDeletion_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(num_star_, gpuptr_position_cur_, position_limits, elimination);

  // zip cur and pre position
  thrust::device_ptr<float2> thrust_position_cur(gpuptr_position_cur_);
  thrust::device_ptr<float2> thrust_position_pre(gpuptr_position_pre_);
  thrust::device_ptr<float>  thrust_weight(gpuptr_weight_);
  thrust::zip_iterator<thrust::tuple<thrust::device_ptr<float2>, thrust::device_ptr<float2>, thrust::device_ptr<float>>>
  position_iter(thrust::make_tuple(thrust_position_cur, thrust_position_pre, thrust_weight));

  // Perform elimination
  thrust::device_ptr<int>    thrust_elimination(elimination);
  thrust::sort_by_key(thrust_elimination, thrust_elimination + num_star_, position_iter);
  num_star_ = thrust::find(thrust_elimination, thrust_elimination + num_star_, 1) - thrust_elimination;

  cudaFree(elimination);
}

/// @endcond
