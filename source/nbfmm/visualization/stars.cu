////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/visualization/stars.cu
/// @brief   The implementation of stars class
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang <davidzan830@gmail.com>
///

/// @cond

#include <nbfmm/visualization.hpp>
#include <cstdint>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <nbfmm/utility.hpp>
#include <nbfmm/model.hpp>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/sort.h>

__global__ void update_kernel(int n_star,float2* gpu_star_position_cur,float2* gpu_star_position_pre,float2* gpu_star_acceleration, float dt)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float2 cur_position = gpu_star_position_cur[idx];

    if ( idx>=n_star )
    {
     return;
    }

    gpu_star_position_cur[idx] = 2*cur_position - gpu_star_position_pre[idx] + dt*dt*gpu_star_acceleration[idx];
    gpu_star_position_pre[idx] = cur_position;
}

__global__ void visualize_kernel(int n_star,float2* gpu_star_position_cur, uint8_t *board,int width, int height,float* gpu_star_weight,float size_th,float4 visualization_limits)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=n_star )
    {
     return;
    }
    float widthUnit=(visualization_limits.z - visualization_limits.x) /width;
    float heightUnit=(visualization_limits.w - visualization_limits.y) /height;
    int pixx=floor((gpu_star_position_cur[idx].x-visualization_limits.x)/widthUnit);
    int pixy=height-1-floor((gpu_star_position_cur[idx].y-visualization_limits.y)/heightUnit);
    int size =floor(gpu_star_weight[idx]/size_th)+1;
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

__global__ void setup_kernel(curandState *state, int n_star)
{

  int idx = threadIdx.x+blockDim.x*blockIdx.x;

  if ( idx>=n_star )
  {
   return;
  }
  curand_init(5678, idx, 0, &state[idx]);
}

__global__ void deletion_check_kernel(int n_star, float2* gpu_star_position_cur, float4 position_limits, int* elimination)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=n_star )
    {
     return;
    }
    if (gpu_star_position_cur[idx].x<position_limits.x || gpu_star_position_cur[idx].y<position_limits.y ||gpu_star_position_cur[idx].x>position_limits.z ||gpu_star_position_cur[idx].y>position_limits.w)
    {
      elimination[idx] = 1;
    } else {
      elimination[idx] = 0;
    }
}
__global__ void initialization_kernel(int n_star,float2* gpu_star_position_cur,float2* gpu_star_position_pre,float2* gpu_star_acceleration,float* gpu_star_weight,float4 position_limits,curandState* d_state, float dt)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = d_state[idx];
  float2 position_pre;
  position_pre.x = position_limits.x+curand_uniform(&localState)*(position_limits.z-position_limits.x-2)+1;
  position_pre.y = position_limits.y+curand_uniform(&localState)*(position_limits.w-position_limits.y-2)+1;

    if ( idx>=n_star )
    {
     return;
    }
      gpu_star_position_cur[idx] = position_pre;
      gpu_star_position_pre[idx] = position_pre;
      gpu_star_acceleration[idx].x = 0;
      gpu_star_acceleration[idx].y = 0;
      gpu_star_weight[idx]=curand_uniform(&localState)*5;

}
// Constructor
Stars::Stars(int nStar, int FPS)
 : n_star(nStar), FPS(FPS), dt(0.05f/FPS)
 {
  assert(nStar>0);
  cudaMalloc(&gpu_star_position_cur,  n_star*sizeof(float2));
  cudaMalloc(&gpu_star_position_pre,  n_star*sizeof(float2));
  cudaMalloc(&gpu_star_acceleration,  n_star*sizeof(float2));
  cudaMalloc(&gpu_star_weight,n_star*sizeof(float));
 }
//Destructor
Stars::~Stars()
{
  cudaFree(gpu_star_position_cur);
  cudaFree(gpu_star_position_pre);
  cudaFree(gpu_star_acceleration);
  cudaFree(gpu_star_weight);
}

void Stars::initialize(float4 position_limit)
{
  // curandState *d_state;
  // cudaMalloc(&d_state,n_star* sizeof(curandState));

  // const int kNumThread_pointwise = 1024;
  // const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  // setup_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(d_state,n_star);
  // initialization_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position_cur,gpu_star_position_pre,gpu_star_acceleration,gpu_star_weight,position_limit,d_state, dt);
  // cudaFree(d_state);

  // const float2 center_position = make_float2((position_limit.x + position_limit.z)/2,
  //                                            (position_limit.y + position_limit.w)/2);
  // const float radius = (position_limit.w - position_limit.y)/64;
  // nbfmm::generateModelDisk(
  //     n_star, center_position, radius, 3.0f, dt, gpu_star_position_cur, gpu_star_position_pre, gpu_star_weight
  // );

  const float2 center_position1 = make_float2(5*position_limit.x + 3*position_limit.z,
                                              5*position_limit.y + 3*position_limit.w)/8;
  const float2 center_position2 = make_float2(3*position_limit.x + 5*position_limit.z,
                                              3*position_limit.y + 5*position_limit.w)/8;
  const float radius = (position_limit.w - position_limit.y)/64;

  nbfmm::generateModelDoubleDisk(
      (n_star*5)/8, (n_star*3)/8, center_position1, center_position2, (radius*5)/8, (radius*3)/8, 3.0f, dt,
      gpu_star_position_cur, gpu_star_position_pre, gpu_star_weight
  );

  // nbfmm::generateModelDoubleDiskCenter(
  //     (n_star*5)/8, (n_star*3)/8, center_position1, center_position2, (radius*5)/8, (radius*3)/8,
  //     1.0f, (n_star*5)/8/256, (n_star*3)/8/256, dt, gpu_star_position_cur, gpu_star_position_pre, gpu_star_weight
  // );
}
//update
void Stars::update()
{
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  update_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position_cur,gpu_star_position_pre,gpu_star_acceleration,dt);
}

void Stars::visualize(int width, int height, uint8_t *board,float size_th,float4  visualization_limits)
{
  cudaMemset(board, 0, width*height);
  cudaMemset(board+width*height, 128, width*height/2);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  visualize_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position_cur,board,width,height,gpu_star_weight, size_th, visualization_limits);
}

void Stars::deletion_check(float4 position_limits)
{
  int* elimination;
  cudaMalloc(&elimination, sizeof(int)*n_star);
  const int kNumThread_pointwise = 1024;
  const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  deletion_check_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star, gpu_star_position_cur, position_limits, elimination);

  // zip cur and pre position
  thrust::device_ptr<float2> thrust_position_cur(gpu_star_position_cur);
  thrust::device_ptr<float2> thrust_position_pre(gpu_star_position_pre);
  thrust::device_ptr<float>  thrust_weight(gpu_star_weight);
  thrust::zip_iterator<thrust::tuple<thrust::device_ptr<float2>, thrust::device_ptr<float2>, thrust::device_ptr<float>>>
  position_iter(thrust::make_tuple(thrust_position_cur, thrust_position_pre, thrust_weight));

  // Perform elimination
  thrust::device_ptr<int>    thrust_elimination(elimination);
  thrust::sort_by_key(thrust_elimination, thrust_elimination + n_star, position_iter);
  n_star = thrust::find(thrust_elimination, thrust_elimination + n_star, 1) - thrust_elimination;

  cudaFree(elimination);
}

/// @endcond
