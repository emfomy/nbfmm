////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/m2m.cu
/// @brief   Compute multipole to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute multipole to multipole
///
/// @param[in]   base_dim       the number of cells in the base level per side.
/// @param[out]  cell_position  the cell positions.
/// @param[out]  cell_weight    the cell weights.
///
__global__ void m2mDevice(
    const int base_dim,
    float2*   cell_position,
    float*    cell_weight
) {
  const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx, idx_above = idx_x + idx_y * blockDim.y * gridDim.y;
  for ( int cell_size = 1; cell_size < base_dim; cell_size *= 2 ) {
    idx = idx_above;
    idx_above += base_dim * base_dim;
    if ( (idx_x % cell_size == 0) && (idx_y % cell_size == 0) ) {
      cell_position[idx_above] = cell_position[idx] * cell_weight[idx];
    }
    __syncthreads();
    if ( (idx_x % cell_size == 0) && (idx_y % (cell_size * 2) == 0) ) {
      cell_weight[idx_above] = cell_weight[idx] + cell_weight[idx+base_dim];
      cell_position[idx_above] += cell_position[idx_above+base_dim];
    }
    __syncthreads();
    if ( (idx_x % (cell_size * 2) == 0) && (idx_y % (cell_size * 2) == 0) ) {
      cell_weight[idx_above]   += cell_weight[idx+1];
      cell_position[idx_above] += cell_position[idx_above+1];
      cell_position[idx_above] /= cell_weight[idx_above];
    }
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
  const int block_dim_side = (base_dim_ < kMaxBlockDim) ? base_dim_ : kMaxBlockDim;
  const int grid_dim_side  = (base_dim_ < kMaxBlockDim) ? 1 : (base_dim_ / block_dim_side);
  const dim3 block_dim(block_dim_side, block_dim_side);
  const dim3 grid_dim(grid_dim_side, grid_dim_side);
  m2mDevice<<<block_dim, grid_dim>>>(base_dim_, gpuptr_cell_position_, gpuptr_cell_weight_);
}

}  // namespace nbfmm
