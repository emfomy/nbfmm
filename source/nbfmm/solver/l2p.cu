////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/l2p.cu
/// @brief   Compute local to particle
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute local to local
///
/// @param[in]      num_level    the number of cell levels.
/// @param[in]      base_size    the number of girds in the base level per side.
/// @param[in/out]  cell_effect  the cell effects.
///
__global__ void l2lDevice(
    const int num_level,
    const int base_size,
    float2*   cell_effect
) {
  const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx   = idx_x + idx_y * base_size + num_level * base_size * base_size;
  for ( int cell_size = base_size; cell_size >= 2; cell_size /= 2 ) {
    int idx_above = idx;
    idx -= base_size * base_size;
    if ( (idx_x % cell_size == 0) && (idx_y % cell_size == 0) ) {
      cell_effect[idx].x             += cell_effect[idx_above].x;
      cell_effect[idx+1].x           += cell_effect[idx_above].x;
      cell_effect[idx+base_size].x   += cell_effect[idx_above].x;
      cell_effect[idx+base_size+1].x += cell_effect[idx_above].x;
      cell_effect[idx].y             += cell_effect[idx_above].y;
      cell_effect[idx+1].y           += cell_effect[idx_above].y;
      cell_effect[idx+base_size].y   += cell_effect[idx_above].y;
      cell_effect[idx+base_size+1].y += cell_effect[idx_above].y;
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute local to particle
///
/// @param[in]      num_particle  the number of particles.
/// @param[in]      base_size     the number of girds in the base level per side.
/// @param[in]      index         the particle cell indices.
/// @param[in]      cell_effect   the cell effects.
/// @param[in/out]  effect        the particle effects.
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
  effect[idx].x += cell_effect[cell_idx].x;
  effect[idx].y += cell_effect[cell_idx].y;
}

//  The namespace NBFMM
namespace nbfmm {

// L2P
void Solver::l2p( const int num_particle ) {
  {
    if ( num_level_ <= 1 ) {
      return;
    }
    const int block_dim_side = (base_size_ < kMaxBlockDim) ? base_size_ : kMaxBlockDim;
    const int grid_dim_side  = (base_size_ < kMaxBlockDim) ? 1 : (base_size_ / block_dim_side);
    assert(grid_dim_side <= kMaxGridDim);
    const dim3 block_dim(block_dim_side, block_dim_side);
    const dim3 grid_dim(grid_dim_side, grid_dim_side);
    l2lDevice<<<block_dim, grid_dim>>>(num_level_, base_size_, gpuptr_cell_effect_);
  }
  {
    if ( num_level_ <= 0 ) {
      return;
    }
    const int block_dim = kMaxBlockDim;
    const int grid_dim  = ((num_particle-1)/block_dim)+1;
    assert(grid_dim <= kMaxGridDim);
    l2pDevice<<<block_dim, grid_dim>>>(num_particle, base_size_, gpuptr_index_, gpuptr_cell_effect_, gpuptr_effect_);
  }
}

}  // namespace nbfmm
