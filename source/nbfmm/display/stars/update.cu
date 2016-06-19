////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars/update.cu
/// @brief   Update the stars
///
/// @author  Mu Yang       <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang  <davidzan830@gmail.com>
///

#include <nbfmm/display/stars.hpp>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_display
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Update the stars
///
/// @param[in]   num_star      the number of stars.
/// @param[in]   tick          the step size in time.
/// @param[out]  position_cur  the current star positions.
/// @param[out]  position_pre  the previous star positions.
/// @param[out]  acceleration  the accelerations.
///

__global__ void updateDevice(
    const int   num_star,
    const float tick,
    float2*     position_cur,
    float2*     position_pre,
    float2*     acceleration
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( idx >= num_star ) {
    return;
  }

  auto position = position_cur[idx];
  position_cur[idx] = 2 * position - position_pre[idx] + acceleration[idx] * tick * tick;
  position_pre[idx] = position;
}

/// @}

// Update the stars
void nbfmm::Stars::update() {
  solver_.solve(num_star_, gpuptr_position_cur_, gpuptr_weight_, gpuptr_acceleration_);

  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_star_-1)/block_dim)+1;
  updateDevice<<<grid_dim, block_dim>>>(num_star_, tick_, gpuptr_position_cur_, gpuptr_position_pre_, gpuptr_acceleration_);

  prune();
}
