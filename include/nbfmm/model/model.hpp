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
//  The namespace NBFMM.
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate uniform circle shape points
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight_current     the device pointer of particle weights.
///
void generateModelCircleUniform(
    const int     num_particles,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight_current
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate circle shape points
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight_current     the device pointer of particle weights.
///
void generateModelCircle(
    const int     num_particles,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight_current
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate disk shape points
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight_current     the device pointer of particle weights.
///
void generateModelDisk(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight_current
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate static disk shape points
///
/// @param[in]   num_particle              the number of particles.
/// @param[in]   center_position           the center position.
/// @param[in]   radius                    the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight_current     the device pointer of particle weights.
///
void generateModelDiskStatic(
    const int     num_particle,
    const float2  center_position,
    const float   radius,
    const float   weight,
    const float   tick,
    float2*       gpuptr_position_current,
    float2*       gpuptr_position_previous,
    float*        gpuptr_weight_current
);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate double disk shape points
///
/// @param[in]   num_particle1             the number of particles.
/// @param[in]   num_particle2             the number of particles.
/// @param[in]   center_position1          the center position.
/// @param[in]   center_position2          the center position.
/// @param[in]   radius1                   the radius.
/// @param[in]   radius2                   the radius.
/// @param[in]   weight                    the weight.
/// @param[in]   tick                      the step size in time.
/// @param[out]  gpuptr_position_current   the device pointer of current particle positions.
/// @param[out]  gpuptr_position_previous  the device pointer of previous particle positions.
/// @param[out]  gpuptr_weight_current     the device pointer of particle weights.
///
void generateModelDoubleDisk(
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
);

}  // namespace nbfmm

#endif  // NBFMM_MODEL_MODEL_HPP_
