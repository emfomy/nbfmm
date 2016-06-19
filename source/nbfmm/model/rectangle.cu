////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/model/rectangle.cu
/// @brief   The implementation of rectangle shape generator.
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/model.hpp>
#include <cstdlib>
#include <cmath>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <nbfmm/core/kernel_function.hpp>
#include <nbfmm/utility.hpp>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate rectangle shape particles
///
/// @param[in]   num_particle       the number of particles.
/// @param[in]   center_position    the center position.
/// @param[in]   width              the width.
/// @param[in]   height             the height.
/// @param[in]   max_weight         the maximum weight.
/// @param[in]   tick               the step size in time.
/// @param[out]  position_current   the current particle positions.
/// @param[out]  position_previous  the previous particle positions.
/// @param[out]  weight             the particle weights.
///
__global__ void generateModelRectangleDevice(
    const int     num_particle,
    const float2  center_position,
    const float   width,
    const float   height,
    const float   max_weight,
    const float   tick,
    float2*       position_current,
    float2*       position_previous,
    float*        weight
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx >= num_particle ) {
    return;
  }

  curandState s;
  curand_init(0, idx, 0, &s);

  const float2 position = center_position + make_float2(curand_uniform(&s) * width  - width/2,
                                                        curand_uniform(&s) * height - height/2);
  position_current[idx]  = center_position;
  position_previous[idx] = center_position;
  weight[idx]            = max_weight * curand_uniform(&s);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM.
//
namespace nbfmm {

// Generate rectangle shape particles
void generateModelRectangle(
    const int     num_particle,
    const float2  center_position,
    const float   width,
    const float   height,
    const float   max_weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
) {
  assert( num_particle > 0 );
  assert( width > 0 );
  assert( height > 0 );
  assert( max_weight > 0 );

  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;

  generateModelRectangleDevice<<<grid_dim, block_dim>>>(num_particle, center_position, width, height, max_weight, tick,
                                                        gpuptr_position_current, gpuptr_position_previous, gpuptr_weight);
}

}
