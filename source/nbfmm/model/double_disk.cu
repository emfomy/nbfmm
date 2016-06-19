////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/model/double_disk.cu
/// @brief   The implementation of double disk shape generator.
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/model.hpp>
#include <cmath>
#include <curand_kernel.h>
#include <nbfmm/core/kernel_function.hpp>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_model
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate double disk shape particles
///
/// @param[in]   num_particle       the number of particles.
/// @param[in]   offset              the offset of previous particle positions.
/// @param[out]  position_previous  the previous particle positions.
///
__global__ void generateDoubleDiskDevice(
    const int  num_particle,
    float2     offset,
    float2*    position_previous
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx >= num_particle ) {
    return;
  }
  position_previous[idx] += offset;
}

/// @}

// Generate double disk shape particles
void nbfmm::model::generateDoubleDisk(
    const int     num_particle1,
    const int     num_particle2,
    const float2  center_position1,
    const float2  center_position2,
    const float   radius1,
    const float   radius2,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight_current
) {
  generateDisk(num_particle1, center_position1, radius1, weight, tick,
                    gpuptr_position_current, gpuptr_position_previous, gpuptr_weight_current);
  generateDisk(num_particle2, center_position2, radius2, weight, tick,
                    gpuptr_position_current+num_particle1, gpuptr_position_previous+num_particle1,
                    gpuptr_weight_current+num_particle1);

  const float2 effect1 = kernelFunction(center_position1, center_position2, weight * num_particle2);
  const float2 effect2 = kernelFunction(center_position2, center_position1, weight * num_particle1);

  float2 distance = center_position1 - center_position2;
  float r = sqrt(distance.x * distance.x + distance.y * distance.y);
  float a1 = sqrt(effect1.x * effect1.x + effect1.y * effect1.y);
  float a2 = sqrt(effect2.x * effect2.x + effect2.y * effect2.y);
  float r1 = r * num_particle2 / (num_particle1 + num_particle2);
  float r2 = r * num_particle1 / (num_particle1 + num_particle2);

  float2 offset1;
  offset1.x = -effect1.y; offset1.y = effect1.x;
  offset1 *= sqrt(r1/a1) * tick;

  float2 offset2;
  offset2.x = -effect2.y; offset2.y = effect2.x;
  offset2 *= sqrt(r2/a2) * tick;

  generateDoubleDiskDevice<<<kMaxBlockDim, ((num_particle1-1)/kMaxBlockDim)+1>>>(
      num_particle1, offset1, gpuptr_position_previous
  );
  generateDoubleDiskDevice<<<kMaxBlockDim, ((num_particle2-1)/kMaxBlockDim)+1>>>(
      num_particle2, offset2, gpuptr_position_previous+num_particle1
  );
}
