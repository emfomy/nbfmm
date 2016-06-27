////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/model/double_disk_center.cu
/// @brief   The implementation of the generator for double disk shape particles with a large particle at each center
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/model.hpp>
#include <cmath>
#include <nbfmm/core/kernel_function.hpp>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_model
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate double disk shape particles with a large particle at each center
///
/// @param[in]   num_particle       the number of particles.
/// @param[in]   offset              the offset of previous particle positions.
/// @param[out]  position_previous  the previous particle positions.
///
__global__ void generateDoubleDiskCenterDevice(
    const int num_particle,
    float2    offset,
    float2*   position_previous
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx >= num_particle ) {
    return;
  }
  position_previous[idx] += offset;
}

/// @}

// Generate double disk shape particles with a large particle at each center
void nbfmm::model::generateDoubleDiskCenter(
    const int    num_particle1,
    const int    num_particle2,
    const float2 center_position1,
    const float2 center_position2,
    const float  radius1,
    const float  radius2,
    const float  weight,
    const float  center_weight1,
    const float  center_weight2,
    const float  eccentricity,
    const float  tick,
    float2*      gpuptr_position_current,
    float2*      gpuptr_position_previous,
    float*       gpuptr_weight_current
) {
  generateDiskCenter(num_particle1, center_position1, radius1, weight, center_weight1, tick,
                    gpuptr_position_current, gpuptr_position_previous, gpuptr_weight_current);
  generateDiskCenter(num_particle2, center_position2, radius2, weight, center_weight2, tick,
                    gpuptr_position_current+num_particle1, gpuptr_position_previous+num_particle1,
                    gpuptr_weight_current+num_particle1);

  const float weight1 = weight * (num_particle1-1) + center_weight1;
  const float weight2 = weight * (num_particle2-1) + center_weight2;

  const float2 effect1 = kernelFunction(center_position1, center_position2, weight2);
  const float2 effect2 = kernelFunction(center_position2, center_position1, weight1);

  float2 distance = center_position1 - center_position2;
  float r = sqrt(distance.x * distance.x + distance.y * distance.y);
  float a1 = sqrt(effect1.x * effect1.x + effect1.y * effect1.y);
  float a2 = sqrt(effect2.x * effect2.x + effect2.y * effect2.y);
  float r1 = r * weight2 / (weight1 + weight2);
  float r2 = r * weight1 / (weight1 + weight2);

  float2 offset1;
  offset1.x = -effect1.y; offset1.y = effect1.x;
  offset1 *= sqrt(r1/a1) * tick / exp2(eccentricity);
  offset1 -= effect1 * tick * tick * eccentricity;

  float2 offset2;
  offset2.x = -effect2.y; offset2.y = effect2.x;
  offset2 *= sqrt(r2/a2) * tick / exp2(eccentricity);
  offset2 -= effect2 * tick * tick * eccentricity;

  generateDoubleDiskCenterDevice<<<kMaxBlockDim, ((num_particle1-1)/kMaxBlockDim)+1>>>(
      num_particle1, offset1, gpuptr_position_previous
  );
  generateDoubleDiskCenterDevice<<<kMaxBlockDim, ((num_particle2-1)/kMaxBlockDim)+1>>>(
      num_particle2, offset2, gpuptr_position_previous+num_particle1
  );
}
