////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/model/circle.cu
/// @brief   The implementation of circle shape generator.
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/model.hpp>
#include <cmath>
#include <curand_kernel.h>
#include <nbfmm/core/kernel_function.hpp>
#include <nbfmm/utility.hpp>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate circle shape particles
///
/// @param[in]   num_particle       the number of particles.
/// @param[in]   center_position    the center position.
/// @param[in]   radius             the radius.
/// @param[in]   weight             the total weight.
/// @param[in]   angle_difference   the difference between current angle and previous angle
/// @param[out]  position_current   the current particle positions.
/// @param[out]  position_previous  the previous particle positions.
/// @param[out]  weight_ptr         the particle weights.
///
__global__ void generateModelCircleDevice(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   angle_difference,
    float2*       position_current,
    float2*       position_previous,
    float*        weight_ptr
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx >= num_particle ) {
    return;
  }
  curandState s;
  curand_init(0, idx, 0, &s);

  const float  angle_current  = 2.0f * M_PI * curand_uniform(&s);
  const float  angle_previous = angle_current - angle_difference;
  position_current[idx]       = center_position + radius * make_float2(cosf(angle_current),  sinf(angle_current));
  position_previous[idx]      = center_position + radius * make_float2(cosf(angle_previous), sinf(angle_previous));
  weight_ptr[idx]             = weight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM.
//
namespace nbfmm {

// Generate circle shape particles
void generateModelCircle(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
) {
  assert( num_particle > 0 );
  assert( radius > 0 );
  assert( weight > 0 );

  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;

  const float2 effect = kernelFunction(make_float2(0.0f, 0.0f), make_float2(radius, 0.0f), weight * (num_particle-1));
  const float angle_difference = acos(1.0 - effect.x * tick * tick / radius / 2.0f);

  generateModelCircleDevice<<<grid_dim, block_dim>>>(num_particle, center_position, radius, weight, angle_difference,
                                                           gpuptr_position_current, gpuptr_position_previous, gpuptr_weight);
}

}
