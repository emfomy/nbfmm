////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars/initialize.cu
/// @brief   Initialize the stars: apply gravitational constant on star weights
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
/// @param[in]   num_star    the number of stars.
/// @param[in]   grav_const  the gravitational constant
/// @param[out]  weight      the star weights.
///

__global__ void initializeDevice(
    const int   num_star,
    const float grav_const,
    float*      weight
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( idx >= num_star ) {
    return;
  }

  weight[idx] *= grav_const;
}

/// @}

// Initialize the stars: apply gravitational constant on star weights
void nbfmm::Stars::initialize() {
  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_star_-1)/block_dim)+1;
  initializeDevice<<<grid_dim, block_dim>>>(num_star_, grav_const_, gpuptr_weight_);
}
