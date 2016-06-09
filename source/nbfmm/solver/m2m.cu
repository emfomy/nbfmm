////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/m2m.cu
/// @brief   Compute multipole to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>
#include <nbfmm/utility.hpp>

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
  const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx, idx_above = idx_x + idx_y * blockDim.y * gridDim.y;
  for ( int cell_size = 1; cell_size < base_size; cell_size *= 2 ) {
    idx = idx_above;
    idx_above += base_size * base_size;
    if ( (idx_x % cell_size == 0) && (idx_y % cell_size == 0) ) {
      cell_position[idx_above] = cell_position[idx] * cell_weight[idx];
    }
    __syncthreads();
    if ( (idx_x % cell_size == 0) && (idx_y % (cell_size * 2) == 0) ) {
      cell_weight[idx_above] = cell_weight[idx] + cell_weight[idx+base_size];
      cell_position[idx_above] += cell_position[idx_above+base_size];
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
  const int block_dim_side = (base_size_ < kMaxBlockDim) ? base_size_ : kMaxBlockDim;
  const int grid_dim_side  = (base_size_ < kMaxBlockDim) ? 1 : (base_size_ / block_dim_side);
  assert(grid_dim_side <= kMaxGridDim);
  const dim3 block_dim(block_dim_side, block_dim_side);
  const dim3 grid_dim(grid_dim_side, grid_dim_side);
  m2mDevice<<<block_dim, grid_dim>>>(base_size_, gpuptr_cell_position_, gpuptr_cell_weight_);
}

}  // namespace nbfmm
