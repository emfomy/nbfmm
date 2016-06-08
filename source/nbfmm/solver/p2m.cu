////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/p2m.cu
/// @brief   Compute particle to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

//  The namespace NBFMM
namespace nbfmm {

// __global__ void p2m_weighting(int num_particle,float2* gpuptr_position_,float* gpuptr_weight_,float2* p2m_buffer)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if ( idx>=num_particle )
//   {
//     return;
//   }
//   p2m_buffer[idx].x=gpuptr_position_[idx].x*gpuptr_weight_[idx];
//   p2m_buffer[idx].y=gpuptr_position_[idx].y*gpuptr_weight_[idx];
// }

// __global__ void p2m_averaging(int base_size_,float2* gpuptr_cell_position_,float* gpuptr_cell_weight_)
// {
//   int thread2Dpx = blockIdx.x * blockDim.x + threadIdx.x;
//   int thread2Dpy = blockIdx.y * blockDim.y + threadIdx.y;
//   if (thread2Dpx >= base_size_ || thread2Dpy >= base_size_)
//     return;
//   int thread1Dp = thread2Dpy * base_size_ + thread2Dpx;

//    gpuptr_cell_position_[thread1Dp].x=gpuptr_cell_position_[thread1Dp].x/gpuptr_cell_weight_[thread1Dp];
//    gpuptr_cell_position_[thread1Dp].y=gpuptr_cell_position_[thread1Dp].y/gpuptr_cell_weight_[thread1Dp];
// }

// struct EqualInt2 {
//   __host__ __device__ bool operator()( const int2 a, const int2 b ) {
//     return ((a.x==b.x)&&(a.y==b.y));
//   }
// };

// struct PlusInt2 {
//   __host__ __device__ int2 operator()( const int2 a, const int2 b ) {
//     return make_int2(a.x+b.x,a.y+b.y);
//   }
// };

// P2M
void Solver::p2m( const int num_particle ) {
  // const dim3 kNumThread_cellwise = (32,32,1);
  // const dim3 kNumBlock_cellwise  = (((base_size_-1)/kNumThread_cellwise)+1,((base_size_-1)/kNumThread_cellwise)+1,1);
  // const int kNumThread_pointwise = 1024;
  // const int kNumBlock_pointwise  = ((num_particle-1)/kNumThread_pointwise)+1;

  // float2* p2m_buffer;
  // int2* p2m_workingspace;
  // cudaMalloc(&p2m_buffer,      max_num_particle_ * sizeof(float2));
  // cudaMalloc(&p2m_workingspace, base_size_ * base_size_*sizeof(int2));
  // p2m_weighting<<<kNumBlock_pointwise,kNumThread_pointwise>>>(num_particle,gpuptr_position_,gpuptr_weight_,p2m_buffer);

  // thrust::device_ptr<int2> thrust_index(gpuptr_index_);
  // thrust::device_ptr<float2> thrust_position(gpuptr_position_);
  // thrust::device_ptr<float> thrust_weight(gpuptr_weight_);
  // thrust::device_ptr<float2> thrust_weighted(p2m_buffer);
  // thrust::device_ptr<int2> thrust_working(p2m_workingspace);
  // thrust::device_ptr<float2> thrust_cellPos(gpuptr_cell_position_);
  // thrust::device_ptr<float2> thrust_cellWei(gpuptr_cell_weight_);
  // EqualInt2 eqa;
  // PlusInt2 plu;

  // thrust::reduce_by_key(thrust_index, thrust_index + num_particle, thrust_weighted, thrust_working, thrust_cellPos, eqa, plu);
  // thrust::reduce_by_key(thrust_index, thrust_index + num_particle, thrust_weight, thrust_working, thrust_cellWei, eqa, plu);
  // p2m_averaging<<<kNumBlock_cellwise,kNumThread_cellwise>>>(base_size_,gpuptr_cell_position_,gpuptr_cell_weight_);



  /// @todo Implement!
}

}  // namespace nbfmm
