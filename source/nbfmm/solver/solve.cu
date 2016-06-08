////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/solve.cu
/// @brief   Solve system
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <nbfmm/solver.hpp>

//  The namespace NBFMM
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute grid index of each particle
///
/// @param[in]   num_particle            the number of particles.
/// @param[in]   position_limits         the limits of positions. [x_min, y_min, x_max, y_max].
/// @param[in]   grid_size               the size of gird. [width, height].
/// @param[in]   gpuptr_position_origin  the original particle positions.
/// @param[out]  gpuptr_index            the particle grid indices.
///
__global__ void computeParticleIndex(
    const int     num_particle,
    const float4  position_limits,
    const float2  grid_size,
    const float2* gpuptr_position_origin,
    int2*         gpuptr_index
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  gpuptr_index[idx].x = floorf((gpuptr_position_origin[idx].x - position_limits.x) / grid_size.x);
  gpuptr_index[idx].y = floorf((gpuptr_position_origin[idx].y - position_limits.y) / grid_size.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Permute input vectors
///
/// @param[in]   num_particle            the number of particles.
/// @param[in]   gpuptr_perm             the particle permutation indices.
/// @param[in]   gpuptr_position_origin  the original particle positions.
/// @param[in]   gpuptr_weight_origin    the original particle weights.
/// @param[out]  gpuptr_position         the particle positions.
/// @param[out]  gpuptr_weight           the particle weights.
///
__global__ void permuteInputVector(
    const int     num_particle,
    const int*    gpuptr_perm,
    const float2* gpuptr_position_origin,
    const float*  gpuptr_weight_origin,
    float2*       gpuptr_position,
    float*        gpuptr_weight
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  gpuptr_position[idx].x = gpuptr_position_origin[gpuptr_perm[idx]].x;
  gpuptr_position[idx].y = gpuptr_position_origin[gpuptr_perm[idx]].y;
  gpuptr_weight[idx]     = gpuptr_weight_origin[gpuptr_perm[idx]];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Permute output vector
///
/// @param[in]   num_particle            the number of particles.
/// @param[in]   gpuptr_perm             the particle permutation indices.
/// @param[out]  gpuptr_position_origin  the original particle effects.
/// @param[in]   gpuptr_position         the particle effects.
///
__global__ void permuteOutputVector(
    const int     num_particle,
    const int*    gpuptr_perm,
    float2*       gpuptr_effect_origin,
    const float2* gpuptr_effect
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  gpuptr_effect_origin[gpuptr_perm[idx]].x = gpuptr_effect[idx].x;
  gpuptr_effect_origin[gpuptr_perm[idx]].y = gpuptr_effect[idx].y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Extract heads of grid index of each grid
///
/// @param[in]   num_particle   the number of particles.
/// @param[in]   base_size      the number of girds in the base level per side.
/// @param[in]   gpuptr_index   the particle grid indices.
/// @param[out]  gpuptr_head    the starting permutation indices of each grid.
///
__global__ void extractHead(
    const int   num_particle,
    const int   base_size,
    const int2* gpuptr_index,
    int*        gpuptr_head
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx>=num_particle ) {
    return;
  }
  if ( idx == 0 ) {
    gpuptr_head[0] = idx;
    gpuptr_head[base_size * base_size] = num_particle;
  } else if ( gpuptr_index[idx].x == gpuptr_index[idx-1].x && gpuptr_index[idx].y != gpuptr_index[idx-1].y ) {
    gpuptr_head[gpuptr_index[idx].x + gpuptr_index[idx].y*base_size] = idx;
  }
}

#pragma warning
__global__ void copyIndexEffect(
    const int     num_particle,
    const int2*   gpuptr_index,
    float2*       gpuptr_effect_origin
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= num_particle ) {
    return;
  }
  gpuptr_effect_origin[idx].x = gpuptr_index[idx].x;
  gpuptr_effect_origin[idx].y = gpuptr_index[idx].y;
}

struct CompareInt2 {
  __host__ __device__ bool operator()( const int2 a, const int2 b ) {
    return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x);
  }
};

// Solve system
void Solver::solve(
    const int     num_particle,
    const float2* gpuptr_position_origin,
    const float*  gpuptr_weight_origin,
    float2*       gpuptr_effect_origin
) {
  assert(num_particle <= max_num_particle_);


  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((num_particle-1)/kNumThread_pointwise)+1;
  

  const float2 grid_size = make_float2((position_limits_.z - position_limits_.x) / base_size_,
                                       (position_limits_.w - position_limits_.y) / base_size_);
  CompareInt2 cmp;
  thrust::device_ptr<int2> thrust_index(gpuptr_index_);
  thrust::device_ptr<int>  thrust_perm(gpuptr_perm_);

  // Copy input vectors
  cudaMemcpy(gpuptr_position_, gpuptr_position_origin, sizeof(float2) * num_particle, cudaMemcpyDeviceToDevice);
  cudaMemcpy(gpuptr_weight_,   gpuptr_weight_origin,   sizeof(float)  * num_particle, cudaMemcpyDeviceToDevice);

  // Compute grid index of each particle
  computeParticleIndex<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, position_limits_, grid_size, gpuptr_position_, gpuptr_index_);

  // Fill particle permutation vector
  thrust::counting_iterator<int> count_iter(0);
  thrust::copy_n(count_iter, num_particle, thrust_perm);

  // Sort values
  thrust::sort_by_key(thrust_index, thrust_index+num_particle, thrust_perm, cmp);

  // Extract heads of grid index of each grid
  extractHead<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, base_size_, gpuptr_index_, gpuptr_head_);

  // Permute input vectors
  permuteInputVector<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, gpuptr_perm_,
                                                gpuptr_position_origin, gpuptr_weight_origin, gpuptr_position_, gpuptr_weight_);

  // FMM
  p2p(num_particle);
  p2m();
  m2m();
  m2l();
  l2p();

  // Permute output vectors
  permuteOutputVector<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, gpuptr_perm_, gpuptr_effect_origin, gpuptr_effect_);

#pragma warning
  // Copy input vectors
  cudaMemcpy(const_cast<float2*>(gpuptr_position_origin), gpuptr_position_,
             num_particle * sizeof(float2), cudaMemcpyDeviceToDevice);
  cudaMemcpy(const_cast<float*>(gpuptr_weight_origin),    gpuptr_weight_,

             sizeof(float)  * num_particle, cudaMemcpyDeviceToDevice);
  copyIndexEffect<<<kNumBlock_pointwise, kNumThread_pointwise>>>(num_particle, gpuptr_index_, gpuptr_effect_origin);

}

}  // namespace nbfmm
