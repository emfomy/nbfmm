////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/model/model.hpp
/// @brief   The definition of model generators.
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_MODEL_MODEL_HPP_
#define NBFMM_MODEL_MODEL_HPP_

#include <nbfmm/config.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The NBFMM namespace.
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The model namespace.
///
namespace model {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate circle shape particles
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateCircle(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate uniform circle shape particles
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateCircleUniform(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate disk shape particles
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateDisk(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate disk shape particles with a large particle at center
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   center_weight             the weight of center particle.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateDiskCenter(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   center_weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate static disk shape particles
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateDiskStatic(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate double disk shape particles
///
/// @param[in]   num_particle1             the number of particles of disk 1.
/// @param[in]   num_particle2             the number of particles of disk 2.
/// @param[in]   center_position1          the center position of disk 1.
/// @param[in]   center_position2          the center position of disk 2.
/// @param[in]   radius1                   the radius of disk 1.
/// @param[in]   radius2                   the radius of disk 2.
/// @param[in]   weight                    the weight.
/// @param[in]   eccentricity              the eccentricity.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateDoubleDisk(
    const int     num_particle1,
    const int     num_particle2,
    const float2  center_position1,
    const float2  center_position2,
    const float   radius1,
    const float   radius2,
    const float   weight,
    const float   eccentricity,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate double disk shape particles with a large particle at each center
///
/// @param[in]   num_particle1             the number of particles of disk 1.
/// @param[in]   num_particle2             the number of particles of disk 2.
/// @param[in]   center_position1          the center position of disk 1.
/// @param[in]   center_position2          the center position of disk 2.
/// @param[in]   radius1                   the radius of disk 1.
/// @param[in]   radius2                   the radius of disk 2.
/// @param[in]   weight                    the weight.
/// @param[in]   center_weight1            the weight of center particle of disk 1.
/// @param[in]   center_weight2            the weight of center particle of disk 2.
/// @param[in]   eccentricity              the eccentricity.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateDoubleDiskCenter(
    const int     num_particle1,
    const int     num_particle2,
    const float2  center_position1,
    const float2  center_position2,
    const float   radius1,
    const float   radius2,
    const float   weight,
    const float   center_weight1,
    const float   center_weight2,
    const float   eccentricity,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate rectangle shape particles
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   width                     the width.
/// @param[in]   height                    the height.
/// @param[in]   max_weight                the maximum weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight             the device pointer of particle weights.
///
void generateRectangle(
    const int     num_particle,
    const float2  center_position,
    const float   width,
    const float   height,
    const float   max_weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight
);

}  // namespace model

}  // namespace nbfmm

#endif  // NBFMM_MODEL_MODEL_HPP_
