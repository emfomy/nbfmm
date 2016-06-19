////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/l2p.cu
/// @brief   Compute local to particle
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_core
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute local to particle
///
/// @param[in]      num_particle  the number of particles.
/// @param[in]      base_dim      the number of cells in the base level per side.
/// @param[in]      index         the particle cell indices.
/// @param[in]      cell_effect   the cell effects.
/// @param[in,out]  effect        the particle effects.
///
__global__ void l2pDevice(
    const int     num_particle,
    const int     base_dim,
    const int2*   index,
    const float2* cell_effect,
    float2*       effect
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx >= num_particle ) {
    return;
  }
  effect[idx] += cell_effect[index[idx].x + index[idx].y * base_dim];
}

/// @}

// L2P
void nbfmm::Solver::l2p( const int num_particle ) {
  if ( num_level_ <= 0 ) {
    return;
  }

  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;
  l2pDevice<<<block_dim, grid_dim>>>(num_particle, base_dim_, gpuptr_index_, gpuptr_cell_effect_, gpuptr_effect_);
}
