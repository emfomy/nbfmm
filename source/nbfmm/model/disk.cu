////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/model/disk.cu
/// @brief   The implementation of disk shape generator.
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
/// Generate disk shape points
///
/// @param[in]   num_particle       the number of particles.
/// @param[in]   center_position    the center position.
/// @param[in]   max_radius         the radius.
/// @param[in]   weight             the weight.
/// @param[in]   tick               the step size in time.
/// @param[out]  position_current   the current particle positions.
/// @param[out]  position_previous  the previous particle positions.
/// @param[out]  weight_ptr         the particle weights.
///
__global__ void generateModelDiskDevice(
    const int     num_particle,
    const float2  center_position,
    const float   max_radius,
    const float   weight,
    const float   tick,
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

  const float  radius           = (float(idx+1) / num_particle) * max_radius;
  const float2 effect           = nbfmm::kernelFunction(make_float2(0.0f, 0.0f), make_float2(radius, 0.0f), weight * idx);
  const float  angle_difference = acos(1.0 - effect.x * tick * tick / radius / 2.0f);
  const float  angle_current    = 2.0f * M_PI * curand_uniform(&s);
  const float  angle_previous   = angle_current - angle_difference;
  position_current[idx]         = center_position + radius * make_float2(cosf(angle_current),  sinf(angle_current));
  position_previous[idx]        = center_position + radius * make_float2(cosf(angle_previous), sinf(angle_previous));
  weight_ptr[idx]               = weight;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM.
//
namespace nbfmm {

// Generate disk shape points
void generateModelDisk(
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

  generateModelDiskDevice<<<grid_dim, block_dim>>>(num_particle, center_position, radius, weight, tick,
                                                   gpuptr_position_current, gpuptr_position_previous, gpuptr_weight);
}

}
