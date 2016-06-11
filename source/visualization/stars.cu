#ifndef DEMO_STARS_CU
#define DEMO_STARS_CU

#include <nbfmm/visualization/stars.hpp>
#include <curand.h>
#include <nbfmm/utility.hpp>

	__global__ update_kernel(int n_star,float2* gpu_star_position,float2* gpu_star_velocity,float2* gpu_star_acceleration, float FPS)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
  		if ( idx>=n_star )
  		{
  		 return;
  		}
  		gpu_star_position[idx]=gpu_star_position[idx]+gpu_star_velocity[idx]/FPS;
  		gpu_star_velocity[idx]=gpu_star_velocity[idx]+gpu_star_acceleration[idx]/FPS;
	}
	//Constructor
	Stars::Stars(int nStar)
	 : n_star(nStar)
	 {
	 	assert(nStar>0);
	 	cudaMalloc(&gpu_star_position, * n_star*sizeof(float2));
	 	cudaMalloc(&gpu_star_velocity, * n_star*sizeof(float2));
	 	cudaMalloc(&gpu_star_acceleration, * n_star*sizeof(float2));
	 }
	//Destructor
	Stars::~Solver()
	{
		cudaFree(gpu_star_position);
		cudaFree(gpu_star_velocity);
		cudaFree(gpu_star_acceleration);
	}

	void Stars::initialize()
	{

	}
	//update
	void Stars::update(int FPS)
	{
		const int kNumThread_pointwise = 1024;
  		const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  		update_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,gpu_star_velocity,gpu_star_acceleration,(float)FPS);
	}


#endif