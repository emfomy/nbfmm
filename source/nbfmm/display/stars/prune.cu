////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars/prune.cu
/// @brief   Remove out-of-range stars
///
/// @author  Mu Yang       <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang  <davidzan830@gmail.com>
///

#include <nbfmm/display/stars.hpp>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_display
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Remove out-of-range stars
///
/// @param[in]   num_star        the number of stars.
/// @param[in]   display_limits  the limits of display positions. [x_min, y_min, x_max, y_max].
/// @param[in]   position        the star positions.
/// @param[out]  elimination     the elimination tag.
///

__global__ void pruneDevice(
    const int     num_star,
    const float4  position_limits,
    const float2* position,
    bool*         elimination
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( idx >= num_star ) {
    return;
  }

  elimination[idx] = ( position[idx].x < position_limits.x || position[idx].y < position_limits.y ||
                       position[idx].x > position_limits.z || position[idx].y > position_limits.w);
}

/// @}

// Remove out-of-range stars
void nbfmm::Stars::prune() {
  thrust::device_ptr<float2> thrust_position_cur(gpuptr_position_cur_);
  thrust::device_ptr<float2> thrust_position_pre(gpuptr_position_pre_);
  thrust::device_ptr<float>  thrust_weight(gpuptr_weight_);
  thrust::device_ptr<bool>   thrust_elimination(gpuptr_elimination_);

  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_star_-1)/block_dim)+1;
  pruneDevice<<<grid_dim, block_dim>>>(num_star_, position_limits_, gpuptr_position_cur_, gpuptr_elimination_);

  // Perform elimination
  thrust::zip_iterator<thrust::tuple<thrust::device_ptr<float2>, thrust::device_ptr<float2>, thrust::device_ptr<float>>>
      iter(thrust::make_tuple(thrust_position_cur, thrust_position_pre, thrust_weight));
  thrust::sort_by_key(thrust_elimination, thrust_elimination + num_star_, iter);
  num_star_ = thrust::find(thrust_elimination, thrust_elimination + num_star_, 1) - thrust_elimination;
}
