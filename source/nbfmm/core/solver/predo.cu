////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/predo.cu
/// @brief   Do preliminary works
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute cell index of each particle
///
/// @param[in]   num_particle     the number of particles.
/// @param[in]   position_limits  the limits of positions. [x_min, y_min, x_max, y_max].
/// @param[in]   cell_size        the size of gird. [width, height].
/// @param[in]   position_origin  the original particle positions.
/// @param[out]  index            the particle cell indices.
///
__global__ void computeParticleIndex(
    const int     num_particle,
    const float4  position_limits,
    const float2  cell_size,
    const float2* position_origin,
    int2*         index
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  index[idx].x = floorf((position_origin[idx].x - position_limits.x) / cell_size.x);
  index[idx].y = floorf((position_origin[idx].y - position_limits.y) / cell_size.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Extract heads of cell index of each cell
///
/// @param[in]   num_particle  the number of particles.
/// @param[in]   base_size     the number of girds in the base level per side.
/// @param[in]   index         the particle cell indices.
/// @param[out]  head          the starting permutation indices of each cell.
///
__global__ void extractHead(
    const int   num_particle,
    const int   base_size,
    const int2* index,
    int*        head
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx>=num_particle ) {
    return;
  }
  if ( idx == 0 ) {
    head[0] = idx;
    head[base_size * base_size] = num_particle;
  } else if ( index[idx] != index[idx-1] ) {
    head[index[idx].x + index[idx].y*base_size] = idx;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Permute input vectors
///
/// @param[in]   num_particle     the number of particles.
/// @param[in]   perm             the particle permutation indices.
/// @param[in]   position_origin  the original particle positions.
/// @param[in]   weight_origin    the original particle weights.
/// @param[out]  position         the particle positions.
/// @param[out]  weight           the particle weights.
///
__global__ void permuteInputVector(
    const int     num_particle,
    const int*    perm,
    const float2* position_origin,
    const float*  weight_origin,
    float2*       position,
    float*        weight
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  position[idx] = position_origin[perm[idx]];
  weight[idx]   = weight_origin[perm[idx]];
}

//  The namespace NBFMM
namespace nbfmm {

// Solve system
void Solver::predo(
    const int     num_particle,
    const float2* gpuptr_position_origin,
    const float*  gpuptr_weight_origin
) {
  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;
  const float2 cell_size = make_float2((position_limits_.z - position_limits_.x) / base_size_,
                                       (position_limits_.w - position_limits_.y) / base_size_);
  thrust::device_ptr<int2> thrust_index(gpuptr_index_);
  thrust::device_ptr<int>  thrust_perm(gpuptr_perm_);

  // Compute cell index of each particle
  computeParticleIndex<<<grid_dim, block_dim>>>(num_particle, position_limits_, cell_size,
                                                gpuptr_position_origin, gpuptr_index_);

  // Fill particle permutation vector
  thrust::counting_iterator<int> count_iter(0);
  thrust::copy_n(count_iter, num_particle, thrust_perm);

  // Sort values
  thrust::sort_by_key(thrust_index, thrust_index+num_particle, thrust_perm);

  // Extract heads of cell index of each cell
  extractHead<<<grid_dim, block_dim>>>(num_particle, base_size_, gpuptr_index_, gpuptr_head_);

  // Permute input vectors
  permuteInputVector<<<grid_dim, block_dim>>>(num_particle, gpuptr_perm_,gpuptr_position_origin,
                                              gpuptr_weight_origin, gpuptr_position_, gpuptr_weight_);
}

}  // namespace nbfmm
