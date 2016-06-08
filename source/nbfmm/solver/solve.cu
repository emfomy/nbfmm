////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/solve.cu
/// @brief   Solve system
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <nbfmm/solver.hpp>

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
  position[idx].x = position_origin[perm[idx]].x;
  position[idx].y = position_origin[perm[idx]].y;
  weight[idx]     = weight_origin[perm[idx]];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Permute output vector
///
/// @param[in]   num_particle     the number of particles.
/// @param[in]   perm             the particle permutation indices.
/// @param[out]  position_origin  the original particle effects.
/// @param[in]   position         the particle effects.
///
__global__ void permuteOutputVector(
    const int     num_particle,
    const int*    perm,
    float2*       effect_origin,
    const float2* effect
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  effect_origin[perm[idx]].x = effect[idx].x;
  effect_origin[perm[idx]].y = effect[idx].y;
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
  } else if ( index[idx].x == index[idx-1].x && index[idx].y != index[idx-1].y ) {
    head[index[idx].x + index[idx].y*base_size] = idx;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The less than operator of int2
///
struct LessThanInt2 {
  __host__ __device__ bool operator()( const int2 a, const int2 b ) {
    return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x);
  }
};

//  The namespace NBFMM
namespace nbfmm {

// Solve system
void Solver::solve(
    const int     num_particle,
    const float2* gpuptr_position_origin,
    const float*  gpuptr_weight_origin,
    float2*       gpuptr_effect_origin
) {
  assert(num_particle <= max_num_particle_);

  const int kNumThread_pointwise = kMaxBlockDim;
  const int kNumBlock_pointwise  = ((num_particle-1)/kNumThread_pointwise)+1;
  assert(kNumBlock_pointwise <= kMaxGridDim);

  const float2 cell_size = make_float2((position_limits_.z - position_limits_.x) / base_size_,
                                       (position_limits_.w - position_limits_.y) / base_size_);
  LessThanInt2 cmp;
  thrust::device_ptr<int2> thrust_index(gpuptr_index_);
  thrust::device_ptr<int>  thrust_perm(gpuptr_perm_);

  // Copy input vectors
  cudaMemcpy(gpuptr_position_, gpuptr_position_origin, sizeof(float2) * num_particle, cudaMemcpyDeviceToDevice);
  cudaMemcpy(gpuptr_weight_,   gpuptr_weight_origin,   sizeof(float)  * num_particle, cudaMemcpyDeviceToDevice);

  // Compute cell index of each particle
  computeParticleIndex<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, position_limits_, cell_size,
                                                                     gpuptr_position_, gpuptr_index_);

  // Fill particle permutation vector
  thrust::counting_iterator<int> count_iter(0);
  thrust::copy_n(count_iter, num_particle, thrust_perm);

  // Sort values
  thrust::sort_by_key(thrust_index, thrust_index+num_particle, thrust_perm, cmp);

  // Extract heads of cell index of each cell
  extractHead<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, base_size_, gpuptr_index_, gpuptr_head_);

  // Permute input vectors
  permuteInputVector<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, gpuptr_perm_, gpuptr_position_origin,
                                                                    gpuptr_weight_origin, gpuptr_position_, gpuptr_weight_);

  // FMM
  p2p(num_particle);
  p2m(num_particle);
  m2m();
  m2l();
  l2p();

  // Permute output vectors
  permuteOutputVector<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, gpuptr_perm_,
                                                                     gpuptr_effect_origin, gpuptr_effect_);
}

}  // namespace nbfmm
