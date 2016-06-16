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
/// Generate circle like points
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

}  // namespace nbfmm

#endif  // NBFMM_MODEL_MODEL_HPP_
