////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/postdo.cu
/// @brief   Do postliminary works
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Permute output vector
///
/// @param[in]   num_particle     the number of particles.
/// @param[in]   perm             the particle permutation indices.
/// @param[out]  effect_origin    the original particle effects.
/// @param[in]   effect           the particle effects.
///
__global__ void permuteOutputVector(
    const int     num_particle,
    const int*    perm,
    float2*       effect_origin,
    const float2* effect
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  effect_origin[perm[idx]] = effect[idx];
}

//  The namespace NBFMM
namespace nbfmm {

// Solve system
void Solver::postdo(
    const int     num_particle,
    float2*       gpuptr_effect_origin
) {
  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;
  permuteOutputVector<<<grid_dim, block_dim>>>(num_particle, gpuptr_perm_, gpuptr_effect_origin, gpuptr_effect_);
}

}  // namespace nbfmm
