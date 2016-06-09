////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/l2p.cu
/// @brief   Compute local to particle
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute local to particle
///
/// @param[in]  num_particle  the number of particles.
/// @param[in]  base_size     the number of girds in the base level per side.
/// @param[in]  index         the particle cell indices.
/// @param[in]  cell_effect   the cell effects.
/// @param      effect        the particle effects.
///
__global__ void l2pDevice(
    const int     num_particle,
    const int     base_size,
    const int2*   index,
    const float2* cell_effect,
    float2*       effect
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  const int cell_idx = index[idx].x + index[idx].y * base_size;
  effect[idx] += cell_effect[cell_idx];
}

//  The namespace NBFMM
namespace nbfmm {

// L2P
void Solver::l2p( const int num_particle ) {
  if ( num_level_ <= 0 ) {
    return;
  }
  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;
  l2pDevice<<<block_dim, grid_dim>>>(num_particle, base_size_, gpuptr_index_, gpuptr_cell_effect_, gpuptr_effect_);
}

}  // namespace nbfmm
