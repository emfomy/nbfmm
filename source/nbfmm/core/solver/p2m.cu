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
/// @param[out]  p2m_buffer       the workspace.
///
__global__ void p2m_weighting( int num_particle, float2* gpuptr_position, float* gpuptr_weight, float2* p2m_buffer ) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx>=num_particle )
  {
   return;
  }
  p2m_buffer[idx].x=gpuptr_position[idx].x*gpuptr_weight[idx];
  p2m_buffer[idx].y=gpuptr_position[idx].y*gpuptr_weight[idx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute particle to multipole averaging
///
/// @param[in]   base_dim              the number of cells in the base level per side.
/// @param[out]  gpuptr_cell_position  the cell positions.
/// @param[in]   gpuptr_cell_weight    the cell positions.
///

__global__ void p2m_assigning( int base_dim, float2* gpuptr_cell_position, float* gpuptr_cell_weight,int assigning_length,int2* p2m_assigningIndex ) {
  int thread2Dpx = blockIdx.x * blockDim.x + threadIdx.x;
  int thread2Dpy = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread2Dpx >= base_dim || thread2Dpy >= base_dim)

   return;
  int thread1Dp = thread2Dpy * base_dim + thread2Dpx;

  gpuptr_cell_position[thread1Dp].x=0;
  gpuptr_cell_position[thread1Dp].y=0;
  gpuptr_cell_weight[thread1Dp]=0;

  if (thread1Dp>=assigning_length)
    return;
  int index_to_assign=p2m_assigningIndex[thread1Dp].y * base_dim +p2m_assigningIndex[thread1Dp].x;
  gpuptr_cell_position[index_to_assign].x=gpuptr_cell_position[base_dim*base_dim+thread1Dp].x/gpuptr_cell_weight[base_dim*base_dim+thread1Dp];
  gpuptr_cell_position[index_to_assign].y=gpuptr_cell_position[base_dim*base_dim+thread1Dp].y/gpuptr_cell_weight[base_dim*base_dim+thread1Dp];
  gpuptr_cell_weight[index_to_assign]=gpuptr_cell_weight[base_dim*base_dim+thread1Dp];
}

//  The namespace NBFMM
namespace nbfmm {

// P2M
void Solver::p2m( const int num_particle ) {
  const dim3 kNumThread_cellwise(32,32,1);
  const dim3 kNumBlock_cellwise(((base_dim_-1)/kNumThread_cellwise.x)+1,((base_dim_-1)/kNumThread_cellwise.y)+1,1);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((num_particle-1)/kNumThread_pointwise)+1;

  float2* p2m_buffer;
  int2* p2m_assigningIndex;
  cudaMalloc(&p2m_buffer,      max_num_particle_ * sizeof(float2));

  cudaMalloc(&p2m_assigningIndex, base_dim_ * base_dim_*sizeof(int2));

  p2m_weighting<<<kNumBlock_pointwise,kNumThread_pointwise>>>(num_particle,gpuptr_position_,gpuptr_weight_,p2m_buffer);

  thrust::device_ptr<int2> thrust_index(gpuptr_index_);
  thrust::device_ptr<float2> thrust_position(gpuptr_position_);
  thrust::device_ptr<float> thrust_weight(gpuptr_weight_);
  thrust::device_ptr<float2> thrust_weighted(p2m_buffer);
  thrust::device_ptr<int2> thrust_assigninging(p2m_assigningIndex);
  thrust::device_ptr<float2> thrust_cellPos(gpuptr_cell_position_);
  thrust::device_ptr<float> thrust_cellWei(gpuptr_cell_weight_);


  thrust::pair<int2*,float*> p2m_dummy;

  thrust::reduce_by_key(thrust_index, thrust_index + num_particle, thrust_weighted, thrust_assigninging, thrust_cellPos+base_dim_ * base_dim_);
  p2m_dummy=thrust::reduce_by_key(thrust_index, thrust_index + num_particle, thrust_weight, thrust_assigninging, thrust_cellWei+base_dim_ * base_dim_);

  int assigning_length=p2m_dummy.second;
  p2m_assigning<<<kNumBlock_cellwise,kNumThread_cellwise>>>(base_dim_,gpuptr_cell_position_,gpuptr_cell_weight_,assigning_length,p2m_assigningIndex);

  cudaFree(p2m_buffer);
  cudaFree(p2m_assigningIndex);

}

}  // namespace nbfmm
