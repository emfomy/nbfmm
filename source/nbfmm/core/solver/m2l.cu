////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/m2l.cu
/// @brief   Compute multipole to local
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute multipole to local
///
/// @param[in]   base_dim             the number of cells in base level per side.
/// @param[in]   level_dim            the number of cells in this level per side.
/// @param[in]   cell_size            the size of cell of this level.
/// @param[in]   cell_level_position  the cell positions of this level.
/// @param[in]   cell_level_weight    the cell weights of this level.
/// @param[out]  cell_level_effect    the cell effects of this level.
///
__global__ void m2lDevice(
    const int     base_dim,
    const int     level_dim,
    const int     cell_size,
    const float2* cell_level_position,
    const float*  cell_level_weight,
    float2*       cell_level_effect
) {
  const int target_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int target_y = threadIdx.y + blockIdx.y * blockDim.y;

  if ( target_x >= level_dim || target_y >= level_dim ) {
    return;
  }

  const int parent_x = target_x & ~1;
  const int parent_y = target_y & ~1;
  const int target_idx = (target_x + target_y * base_dim) * cell_size;
  const float2 target_position = cell_level_position[target_idx];
  float2 target_effect = make_float2(0.0f, 0.0f);

  // Go through children of parent cell's neighbors
  for ( int y = parent_y-2; y < parent_y+4; ++y ) {
    if ( y >= 0 && y < level_dim ) {
      for ( int x = parent_x-2; x < parent_x+4; ++x ) {
        if ( x >= 0 && x < level_dim ) {
          // Ignore target cell's neighbors
          if ( abs(x-target_x) > 1 || abs(y-target_y) > 1 ) {
            int idx = (x + y * base_dim) * cell_size;
            target_effect += nbfmm::kernelFunction(target_position, cell_level_position[idx], cell_level_weight[idx]);
          }
        }
      }
    }
  }
  cell_level_effect[target_idx] = target_effect;
}

//  The namespace NBFMM
namespace nbfmm {

// M2L
void Solver::m2l() {
  if ( num_level_ <= 0 ) {
    return;
  }

  int level_dim = base_dim_;
  int cell_size = 1;
  for ( auto level = 0; level < num_level_; ++level, level_dim /= 2, cell_size *= 2 ) {
    const int block_dim_side = (level_dim < kMaxBlockDim) ? level_dim : kMaxBlockDim;
    const int grid_dim_side  = (level_dim < kMaxBlockDim) ? 1 : (level_dim / block_dim_side);
    const dim3 block_dim(block_dim_side, block_dim_side);
    const dim3 grid_dim(grid_dim_side, grid_dim_side);
    const int offset = level * base_dim_ * base_dim_;
    m2lDevice<<<block_dim, grid_dim>>>(base_dim_, level_dim, cell_size,
                                       gpuptr_cell_position_ + offset, gpuptr_cell_weight_ + offset, gpuptr_cell_effect_ + offset);
  }
}

}  // namespace nbfmm
