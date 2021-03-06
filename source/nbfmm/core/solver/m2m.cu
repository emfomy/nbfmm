////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/m2m.cu
/// @brief   Compute multipole to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_core
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute multipole to multipole
///
/// @param[in]      cell_size       the size of cells in the current level per side.
/// @param[in]      base_dim        the number of cells in the base level per side.
/// @param[in,out]  level_position  the cell positions of current level.
/// @param[in,out]  level_weight    the cell weights of current level.
///
__global__ void m2mDevice(
    const int cell_size,
    const int base_dim,
    float2*   level_position,
    float*    level_weight
) {
  const int thread_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int thread_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

  if ( thread_idx_x >= base_dim / cell_size || thread_idx_y >= base_dim / cell_size ) {
    return;
  }

  const int idx       = (thread_idx_x + thread_idx_y * base_dim) * cell_size;
  const int idx_child = idx - base_dim * base_dim;
  const int offset_x  = cell_size / 2;
  const int offset_y  = offset_x * base_dim;

  const auto this_position = level_position[idx_child]                       * level_weight[idx_child]
                           + level_position[idx_child + offset_x]            * level_weight[idx_child + offset_x]
                           + level_position[idx_child + offset_y]            * level_weight[idx_child + offset_y]
                           + level_position[idx_child + offset_x + offset_y] * level_weight[idx_child + offset_x + offset_y];
  const auto this_weight   = level_weight[idx_child]
                           + level_weight[idx_child + offset_x]
                           + level_weight[idx_child + offset_y]
                           + level_weight[idx_child + offset_x + offset_y];
  if ( this_weight == 0.0f ) {
    level_position[idx] = make_float2(0.0f, 0.0f);
  } else {
    level_position[idx] = this_position / this_weight;
  }
  level_weight[idx] = this_weight;
}

/// @}

// M2M
void nbfmm::Solver::m2m() {
  if ( num_level_ <= 1 ) {
    return;
  }

  for ( auto level = 1; level < num_level_; ++level ) {
    const int cell_size = 1 << level;
    const int level_dim = base_dim_ / cell_size;
    const int block_dim_side = (level_dim < kMaxBlockDim) ? level_dim : kMaxBlockDim;
    const int grid_dim_side  = (level_dim < kMaxBlockDim) ? 1 : (level_dim / block_dim_side);
    const dim3 block_dim(block_dim_side, block_dim_side);
    const dim3 grid_dim(grid_dim_side, grid_dim_side);
    const int offset = level * base_dim_ * base_dim_;
    m2mDevice<<<block_dim, grid_dim>>>(cell_size, base_dim_, gpuptr_cell_position_ + offset, gpuptr_cell_weight_ + offset);
  }
}
