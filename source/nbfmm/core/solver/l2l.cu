////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/l2l.cu
/// @brief   Compute local to local
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute local to local
///
/// @param[in]      cell_size     the size of cells in the current level per side.
/// @param[in]      base_dim      the number of cells in the base level per side.
/// @param[in,out]  level_effect  the cell effects of the current level.
///
__global__ void l2lDevice(
    const int cell_size,
    const int base_dim,
    float2*   level_effect
) {
  const int thread_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int thread_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

  if ( thread_idx_x >= base_dim / cell_size || thread_idx_y >= base_dim / cell_size ) {
    return;
  }

  const int idx          = ( thread_idx_x       +  thread_idx_y       * base_dim) * cell_size;
  const int idx_parent   = ((thread_idx_x & ~1) + (thread_idx_y & ~1) * base_dim) * cell_size + base_dim * base_dim;
  level_effect[idx] += level_effect[idx_parent];
}

//  The namespace NBFMM
namespace nbfmm {

// L2L
void Solver::l2l() {
  if ( num_level_ <= 1 ) {
    return;
  }

  for ( auto level = num_level_-2; level >= 0; --level ) {
    const int cell_size = 1 << level;
    const int level_dim = base_dim_ / cell_size;
    const int block_dim_side = (level_dim < kMaxBlockDim) ? level_dim : kMaxBlockDim;
    const int grid_dim_side  = (level_dim < kMaxBlockDim) ? 1 : (level_dim / block_dim_side);
    const dim3 block_dim(block_dim_side, block_dim_side);
    const dim3 grid_dim(grid_dim_side, grid_dim_side);
    const int offset = level * base_dim_ * base_dim_;
    l2lDevice<<<block_dim, grid_dim>>>(cell_size, base_dim_, gpuptr_cell_effect_ + offset);
  }
}

}  // namespace nbfmm
