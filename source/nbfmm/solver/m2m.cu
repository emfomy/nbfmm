////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/m2m.cu
/// @brief   Compute multipole to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute multipole to multipole
///
/// @param[in]   base_size      the number of girds in the base level per side.
/// @param[out]  cell_position  the cell positions.
/// @param[out]  cell_weight    the cell weights.
///
__global__ void m2mDevice(
    const int base_size,
    float2*   cell_position,
    float*    cell_weight
) {
  const int idx_x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const int idx_y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  int idx   = idx_x + idx_y * base_size;
  for ( int cell_size = 2; cell_size <= base_size; cell_size *= 2 ) {
    int idx_above = idx + base_size * base_size;
    if ( (idx_x % cell_size == 0) && (idx_y % cell_size == 0) ) {
      float weight_above = cell_weight[idx] + cell_weight[idx+1] + cell_weight[idx+base_size] + cell_weight[idx+base_size+1];
      float position_above_x = cell_position[idx].x             * cell_weight[idx]
                             + cell_position[idx+1].x           * cell_weight[idx+1]
                             + cell_position[idx+base_size].x   * cell_weight[idx+base_size]
                             + cell_position[idx+base_size+1].x * cell_weight[idx+base_size+1];
      float position_above_y = cell_position[idx].y             * cell_weight[idx]
                             + cell_position[idx+1].y           * cell_weight[idx+1]
                             + cell_position[idx+base_size].y   * cell_weight[idx+base_size]
                             + cell_position[idx+base_size+1].y * cell_weight[idx+base_size+1];
      cell_weight[idx_above]     = weight_above;
      cell_position[idx_above].x = position_above_x / weight_above;
      cell_position[idx_above].y = position_above_y / weight_above;
    }
    idx = idx_above;
    __syncthreads();
  }
}

//  The namespace NBFMM
namespace nbfmm {

// M2M
void Solver::m2m() {
  if ( num_level_ <= 1 ) {
    return;
  }
  const int base_size_half = base_size_/2;
  const int block_dim_side = (base_size_half < kMaxBlockDim) ? base_size_half : kMaxBlockDim;
  const int grid_dim_side  = (base_size_half < kMaxBlockDim) ? 1 : (base_size_half / block_dim_side);
  assert(grid_dim_side <= kMaxGridDim);
  const dim3 block_dim(block_dim_side, block_dim_side);
  const dim3 grid_dim(grid_dim_side, grid_dim_side);
  m2mDevice<<<block_dim, grid_dim>>>(base_size_, gpuptr_cell_position_, gpuptr_cell_weight_);
}

}  // namespace nbfmm
