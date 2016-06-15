////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/visualization/stars.cu
/// @brief   The implementation of stars class
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
///

/// @cond

#include <nbfmm/visualization.hpp>
#include <cstdint>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <nbfmm/utility.hpp>

__global__ void update_kernel(int n_star,float2* gpu_star_position,float2* gpu_star_velocity,float2* gpu_star_acceleration, float FPS)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=n_star )
    {
     return;
    }
    gpu_star_position[idx]=gpu_star_position[idx]+gpu_star_velocity[idx]/FPS;
    gpu_star_velocity[idx]=gpu_star_velocity[idx]+gpu_star_acceleration[idx]/FPS;
}

__global__ void visualize_kernel(int n_star,float2* gpu_star_position, uint8_t *board,int width, int height,float* gpu_star_weight,float size_th,float4 visualization_limits)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=n_star )
    {
     return;
    }
    float widthUnit=(visualization_limits.z - visualization_limits.x) /width;
    float heightUnit=(visualization_limits.w - visualization_limits.y) /height;
    int pixx=floor((gpu_star_position[idx].x-visualization_limits.x)/widthUnit);
    int pixy=height-1-floor((gpu_star_position[idx].y-visualization_limits.y)/heightUnit);
    int size =floor(gpu_star_weight[idx]/size_th)+1;
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

__global__ void setup_kernel(curandState *state, int n_star)
{

int idx = threadIdx.x+blockDim.x*blockIdx.x;

if ( idx>=n_star )
    {
     return;
    }
curand_init(1234, idx, 0, &state[idx]);
}

__global__ void deletion_check_kernel(int n_star,float2* gpu_star_position,float2* gpu_star_velocity,float2* gpu_star_acceleration,float* gpu_star_weight,float4 position_limits,curandState* d_state)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx>=n_star )
    {
     return;
    }
    if (gpu_star_position[idx].x<position_limits.x || gpu_star_position[idx].y<position_limits.y ||gpu_star_position[idx].x>position_limits.z ||gpu_star_position[idx].y>position_limits.w)
    {
      float tmpx=gpu_star_position[idx].x/(position_limits.z-position_limits.x);
      gpu_star_position[idx].x=(tmpx-floor(tmpx))*(position_limits.z-position_limits.x);
      float tmpy=gpu_star_position[idx].y/(position_limits.w-position_limits.y);
      gpu_star_position[idx].y=(tmpy-floor(tmpy))*(position_limits.w-position_limits.y);
    }
}
__global__ void initialization_kernel(int n_star,float2* gpu_star_position,float2* gpu_star_velocity,float2* gpu_star_acceleration,float* gpu_star_weight,float4 position_limits,curandState* d_state)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = d_state[idx];
    if ( idx>=n_star )
    {
     return;
    }

      gpu_star_position[idx].x=position_limits.x+curand_uniform(&localState)*(position_limits.z-position_limits.x);
      gpu_star_position[idx].y=position_limits.y+curand_uniform(&localState)*(position_limits.w-position_limits.y);
      gpu_star_velocity[idx].x=curand_normal(&localState);
      gpu_star_velocity[idx].y=curand_normal(&localState);
      gpu_star_acceleration[idx].x=0;
      gpu_star_acceleration[idx].y=0;
      gpu_star_weight[idx]=curand_uniform(&localState)*5;

}
//Constructor
Stars::Stars(int nStar)
 : n_star(nStar)
 {
  assert(nStar>0);
  cudaMalloc(&gpu_star_position,  n_star*sizeof(float2));
  cudaMalloc(&gpu_star_velocity,  n_star*sizeof(float2));
  cudaMalloc(&gpu_star_acceleration,  n_star*sizeof(float2));
  cudaMalloc(&gpu_star_weight,n_star*sizeof(float));
 }
//Destructor
Stars::~Stars()
{
  cudaFree(gpu_star_position);
  cudaFree(gpu_star_velocity);
  cudaFree(gpu_star_acceleration);
  cudaFree(gpu_star_weight);
}

void Stars::initialize(float4 position_limit)
{
  curandState *d_state;
    cudaMalloc(&d_state,n_star* sizeof(curandState));

    const int kNumThread_pointwise = 1024;
    const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
    setup_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(d_state,n_star);
    initialization_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,gpu_star_velocity,gpu_star_acceleration,gpu_star_weight,position_limit,d_state);
    cudaFree(d_state);
}
//update
void Stars::update(int FPS)
{
  const int kNumThread_pointwise = 1024;
    const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
    update_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,gpu_star_velocity,gpu_star_acceleration,(float)FPS);
}

void Stars::visualize(int width, int height, uint8_t *board,float size_th,float4  visualization_limits)
{
  cudaMemset(board, 0, width*height);
  cudaMemset(board+width*height, 128, width*height/2);
  const int kNumThread_pointwise = 1024;
    const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
    visualize_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,board,width,height,gpu_star_weight, size_th, visualization_limits);
}

void Stars::deletion_check(float4 position_limits)
{
  curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    const int kNumThread_pointwise = 1024;
    const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
    setup_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(d_state,n_star);
    deletion_check_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,gpu_star_velocity,gpu_star_acceleration,gpu_star_weight,position_limits,d_state);
    cudaFree(d_state);
}

/// @endcond
