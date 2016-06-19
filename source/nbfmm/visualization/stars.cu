////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars.cu
/// @brief   The implementation of stars class
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang <davidzan830@gmail.com>
///

/// @cond

#include <nbfmm/display.hpp>
#include <cstdint>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <nbfmm/utility.hpp>
#include <nbfmm/model.hpp>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/sort.h>

__global__ void update_kernel(int num_star,float2* gpuptr_position_cur,float2* gpuptr_position_pre,float2* gpuptr_acceleration, float tick)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float2 cur_position = gpuptr_position_cur[idx];

    if ( idx>=num_star )
    {
     return;
    }

    gpuptr_position_cur[idx] = 2*cur_position - gpuptr_position_pre[idx] + tick*tick*gpuptr_acceleration[idx];
    gpuptr_position_pre[idx] = cur_position;
}

__global__ void visualize_kernel(int num_star,float2* gpuptr_position_cur, uint8_t *board,int width, int height,float* gpuptr_weight,float size_scale,float4 display_limits)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=num_star )
    {
     return;
    }
    float widthUnit=(display_limits.z - display_limits.x) /width;
    float heightUnit=(display_limits.w - display_limits.y) /height;
    int pixx=floor((gpuptr_position_cur[idx].x-display_limits.x)/widthUnit);
    int pixy=height-1-floor((gpuptr_position_cur[idx].y-display_limits.y)/heightUnit);
    int size =floor(gpuptr_weight[idx]/size_scale)+1;
    if ( 0 <= pixx && pixx < width && 0 <= pixy && pixy <= height )
    {
      if (size>=1)
      {
        board[pixy*width+pixx]=255;
      }
      if (size>=2)
      {
        if (pixy>0)
          board[(pixy-1)*width+pixx]=255;
        if (pixy<height-1)
          board[(pixy+1)*width+pixx]=255;
        if (pixx>0)
          board[pixy*width+(pixx-1)]=255;
        if (pixx<width-1)
          board[pixy*width+(pixx+1)]=255;
      }
      if (size>=3)
      {
        if (pixy>0 && pixx>0)
          board[(pixy-1)*width+(pixx-1)]=255;
        if (pixy<height-1 && pixx>0)
          board[(pixy+1)*width+(pixx-1)]=255;
        if (pixy>0 && pixx<width-1)
          board[(pixy-1)*width+(pixx+1)]=255;
        if (pixy<height-1 && pixx<width-1)
          board[(pixy+1)*width+(pixx+1)]=255;
      }
      if (size>=4)
      {
        if (pixy>1)
          board[(pixy-2)*width+pixx]=255;
        if (pixy<height-2)
          board[(pixy+2)*width+pixx]=255;
        if (pixx>1)
          board[pixy*width+(pixx-2)]=255;
        if (pixx<width-2)
          board[pixy*width+(pixx+2)]=255;
      }
      if (size>=5)
      {
        if (pixy>1 && pixx>0)
          board[(pixy-2)*width+(pixx-1)]=255;
        if (pixy>1 && pixx<width-1)
          board[(pixy-2)*width+(pixx+1)]=255;
        if (pixy<height-2 && pixx>0)
          board[(pixy+2)*width+(pixx-1)]=255;
        if (pixy<height-2 && width-1)
          board[(pixy+2)*width+(pixx+1)]=255;
        if (pixy>0 && pixx<width-2)
          board[(pixy-1)*width+(pixx+2)]=255;
        if (pixy<height-1 && pixx<width-2)
          board[(pixy+1)*width+(pixx+2)]=255;
        if (pixy<height-1 && pixx>1)
          board[(pixy+1)*width+(pixx-2)]=255;
        if (pixy>0 && pixx>1)
          board[(pixy-1)*width+(pixx-2)]=255;
      }
      if (size>=6)
      {
        if (pixy>1 && pixx>1)
          board[(pixy-2)*width+(pixx-2)]=255;
        if (pixy<height-2 && pixx>2)
          board[(pixy+2)*width+(pixx-2)]=255;
        if (pixy>1 && pixx<width-2)
          board[(pixy-2)*width+(pixx+2)]=255;
        if (pixy<height-2 && pixx<width-2)
          board[(pixy+2)*width+(pixx+2)]=255;
      }
    }
}

__global__ void setup_kernel(curandState *state, int num_star)
{

  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if ( idx>=num_star )
  {
   return;
  }
  curand_init(5678, idx, 0, &state[idx]);
}

__global__ void deletion_check_kernel(int num_star, float2* gpuptr_position_cur, float4 position_limits, int* elimination)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=num_star )
    {
     return;
    }
    if (gpuptr_position_cur[idx].x<position_limits.x || gpuptr_position_cur[idx].y<position_limits.y ||gpuptr_position_cur[idx].x>position_limits.z ||gpuptr_position_cur[idx].y>position_limits.w)
    {
      elimination[idx] = 1;
    } else {
      elimination[idx] = 0;
    }
}

void nbfmm::Stars::initialize()
{
  auto position_limits = position_limits_;

  // const float2 center_position = (make_float2(position_limits.x, position_limits.y) +
  //                                 make_float2(position_limits.z, position_limits.w)) / 2;
  // const float width  = (position_limits.z - position_limits.x)/2;
  // const float height = (position_limits.w - position_limits.y)/2;

  // nbfmm::generateModelRectangle(
  //     num_star_, center_position, width, height, 6.0f, tick_, gpuptr_position_cur_, gpuptr_position_pre_, gpuptr_weight_
  // );

  const int n1 = 5;
  const int n2 = 3;
  const float mu1 = float(n1) / (n1+n2);
  const float mu2 = float(n2) / (n1+n2);

  const float2 center_position1 = (make_float2(position_limits.x, position_limits.y) * (3*mu1+2*mu2) +
                                   make_float2(position_limits.z, position_limits.w) * (3*mu1+4*mu2)) / 6;
  const float2 center_position2 = (make_float2(position_limits.z, position_limits.w) * (3*mu2+2*mu1) +
                                   make_float2(position_limits.x, position_limits.y) * (3*mu2+4*mu1)) / 6;
  const float radius = (position_limits.w - position_limits.y)/16;

  // nbfmm::generateModelDisk(
  //     num_star_, center_position1, radius, 3.0f, tick_, gpuptr_position_cur_, gpuptr_position_pre_, gpuptr_weight_
  // );

  nbfmm::generateModelDoubleDisk(
      num_star_*mu1, num_star_*mu2, center_position1, center_position2, radius*mu1, radius*mu2, 3.0f, tick_,
      gpuptr_position_cur_, gpuptr_position_pre_, gpuptr_weight_
  );

  // nbfmm::generateModelDoubleDiskCenter(
  //     num_star_*mu1, num_star_*mu2, center_position1, center_position2, radius*mu1, radius*mu2, 1.0f,
  //     num_star_*mu1, num_star_*mu1, tick_, gpuptr_position_cur_, gpuptr_position_pre_, gpuptr_weight_
  // );
}
//update
void nbfmm::Stars::update()
{
  solver_.solve(num_star_, gpuptr_position_cur_, gpuptr_weight_, gpuptr_acceleration_);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((num_star_-1)/kNumThread_pointwise)+1;
  update_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(num_star_,gpuptr_position_cur_,gpuptr_position_pre_,gpuptr_acceleration_,tick_);

  deletion_check();
}

void nbfmm::Stars::display(uint8_t *board)
{
  auto width = width_;
  auto height = height_;
  auto size_scale = size_scale_;
  auto display_limits = display_limits_;
  cudaMemset(board, 0, width*height);
  cudaMemset(board+width*height, 128, width*height/2);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((num_star_-1)/kNumThread_pointwise)+1;
  visualize_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(num_star_,gpuptr_position_cur_,board,width,height,gpuptr_weight_, size_scale, display_limits);

  deletion_check();
}

void nbfmm::Stars::deletion_check()
{
  auto position_limits = position_limits_;
  int* elimination;
  cudaMalloc(&elimination, sizeof(int)*num_star_);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((num_star_-1)/kNumThread_pointwise)+1;
  deletion_check_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(num_star_, gpuptr_position_cur_, position_limits, elimination);

  // zip cur and pre position
  thrust::device_ptr<float2> thrust_position_cur(gpuptr_position_cur_);
  thrust::device_ptr<float2> thrust_position_pre(gpuptr_position_pre_);
  thrust::device_ptr<float>  thrust_weight(gpuptr_weight_);
  thrust::zip_iterator<thrust::tuple<thrust::device_ptr<float2>, thrust::device_ptr<float2>, thrust::device_ptr<float>>>
  position_iter(thrust::make_tuple(thrust_position_cur, thrust_position_pre, thrust_weight));

  // Perform elimination
  thrust::device_ptr<int>    thrust_elimination(elimination);
  thrust::sort_by_key(thrust_elimination, thrust_elimination + num_star_, position_iter);
  num_star_ = thrust::find(thrust_elimination, thrust_elimination + num_star_, 1) - thrust_elimination;

  cudaFree(elimination);
}

/// @endcond
