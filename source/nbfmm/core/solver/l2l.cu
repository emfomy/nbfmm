////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/l2l.cu
/// @brief   Compute local to local
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute local to local
///
/// @param[in]  num_level    the number of cell levels.
/// @param[in]  base_dim     the number of cells in the base level per side.
/// @param      cell_effect  the cell effects.
///
__global__ void l2lDevice(
    const int num_level,
    const int base_dim,
    float2*   cell_effect
) {
  const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx   = idx_x + idx_y * base_dim + num_level * base_dim * base_dim;
  for ( int cell_size = base_dim; cell_size >= 2; cell_size /= 2 ) {
    int idx_above = idx;
    idx -= base_dim * base_dim;
    if ( (idx_x % cell_size == 0) && (idx_y % cell_size == 0) ) {
      cell_effect[idx].x            += cell_effect[idx_above].x;
      cell_effect[idx+1].x          += cell_effect[idx_above].x;
      cell_effect[idx+base_dim].x   += cell_effect[idx_above].x;
      cell_effect[idx+base_dim+1].x += cell_effect[idx_above].x;
      cell_effect[idx].y            += cell_effect[idx_above].y;
      cell_effect[idx+1].y          += cell_effect[idx_above].y;
      cell_effect[idx+base_dim].y   += cell_effect[idx_above].y;
      cell_effect[idx+base_dim+1].y += cell_effect[idx_above].y;
    }
    __syncthreads();
  }
}

//  The namespace NBFMM
namespace nbfmm {

// L2P
void Solver::l2l() {
  if ( num_level_ <= 1 ) {
    return;
  }
  const int block_dim_side = (base_dim_ < kMaxBlockDim) ? base_dim_ : kMaxBlockDim;
  const int grid_dim_side  = (base_dim_ < kMaxBlockDim) ? 1 : (base_dim_ / block_dim_side);
  const dim3 block_dim(block_dim_side, block_dim_side);
  const dim3 grid_dim(grid_dim_side, grid_dim_side);
  l2lDevice<<<block_dim, grid_dim>>>(num_level_, base_dim_, gpuptr_cell_effect_);
}

}  // namespace nbfmm
