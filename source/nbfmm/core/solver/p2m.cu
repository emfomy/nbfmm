////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/p2m.cu
/// @brief   Compute particle to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute particle to multipole weighting
///
/// @param[in]   num_particle     the number of particles.
/// @param[in]   gpuptr_position  the original particle effects.
/// @param[in]   gpuptr_weight    the particle effects.
/// @param[out]  buffer           the workspace.
///
__global__ void p2mWeight(
    const int     num_particle,
    const float2* gpuptr_position,
    const float*  gpuptr_weight,
    float2*       buffer
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx>=num_particle ) {
   return;
  }
  buffer[idx] = gpuptr_position[idx] * gpuptr_weight[idx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute particle to multipole averaging
///
/// @param[in]   base_dim              the number of cells in the base level per side.
/// @param[in]   buffer_length         the length of workspace.
/// @param[in]   buffer                the workspace.
/// @param[out]  gpuptr_cell_position  the cell positions.
/// @param[out]  gpuptr_cell_weight    the cell positions.
///
__global__ void p2mAssigning(
    const int    base_dim,
    const int    buffer_length,
    const int2*  buffer,
    float2*      gpuptr_cell_position,
    float*       gpuptr_cell_weight
) {
  const int thread2Dpx = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread2Dpy = blockIdx.y * blockDim.y + threadIdx.y;

  if (thread2Dpx >= base_dim || thread2Dpy >= base_dim) {
   return;
  }

  const int thread1Dp = thread2Dpy * base_dim + thread2Dpx;

  gpuptr_cell_position[thread1Dp] = make_float2(0.0f, 0.0f);
  gpuptr_cell_weight[thread1Dp] = 0;

  if ( thread1Dp >= buffer_length ) {
    return;
  }

  const int index_to_assign = buffer[thread1Dp].x + buffer[thread1Dp].y * base_dim;
  const int index_temp      = thread1Dp + base_dim * base_dim;

  gpuptr_cell_position[index_to_assign] = gpuptr_cell_position[index_temp] / gpuptr_cell_weight[index_temp];
  gpuptr_cell_weight[index_to_assign]   = gpuptr_cell_weight[index_temp];
}

//  The namespace NBFMM
namespace nbfmm {

// P2M
void Solver::p2m( const int num_particle ) {
  if ( num_level_ <= 0 ) {
    return;
  }

  const int block_dim_particle = kMaxBlockDim;
  const int grid_dim_particle  = ((num_particle-1)/block_dim_particle)+1;

  const int block_dim_side = 32;
  const int grid_dim_side  = ((base_dim_-1)/block_dim_side)+1;
  const dim3 block_dim_cell(block_dim_side, block_dim_side);
  const dim3 grid_dim_cell(grid_dim_side, grid_dim_side);

  p2mWeight<<<grid_dim_particle, block_dim_particle>>>(num_particle, gpuptr_position_, gpuptr_weight_, gpuptr_buffer_float2_);

  thrust::device_ptr<int2>   thrust_index(gpuptr_index_);
  thrust::device_ptr<float2> thrust_position(gpuptr_position_);
  thrust::device_ptr<float>  thrust_weight(gpuptr_weight_);
  thrust::device_ptr<float2> thrust_weighted(gpuptr_buffer_float2_);
  thrust::device_ptr<int2>   thrust_assigninging(gpuptr_buffer_int2_);
  thrust::device_ptr<float2> thrust_cellPos(gpuptr_cell_position_);
  thrust::device_ptr<float>  thrust_cellWei(gpuptr_cell_weight_);

  thrust::reduce_by_key(thrust_index, thrust_index + num_particle, thrust_weighted,
                        thrust_assigninging, thrust_cellPos + base_dim_ * base_dim_);
  auto p2m_dummy = thrust::reduce_by_key(thrust_index, thrust_index + num_particle, thrust_weight,
                                         thrust_assigninging, thrust_cellWei + base_dim_ * base_dim_);

  const int buffer_int2_length = p2m_dummy.second - (thrust_cellWei + base_dim_ * base_dim_);
  p2mAssigning<<<grid_dim_cell, block_dim_cell>>>(base_dim_, buffer_int2_length, gpuptr_buffer_int2_,
                                                  gpuptr_cell_position_, gpuptr_cell_weight_);
}

}  // namespace nbfmm
