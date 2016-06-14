#ifndef DEMO_STARS_CU
#define DEMO_STARS_CU

#include <nbfmm/visualization/stars.hpp>
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
  		float widthUnit=(position_limits_.z - position_limits_.x) /width;
  		float heightUnit=(position_limits_.w - position_limits_.y) /height;
  		int pixx=floor((gpu_star_position[idx].x-position_limits_.x)/widthUnit);
  		int pixy=height-1-floor((gpu_star_position[idx].y-position_limits_.y)/heighthUnit);
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

	__global__ void setup_kernel(curandState *state)
	{

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
	}

	__global__ void deletion_check_kernel(int n_star,float2* gpu_star_position,float2* gpu_star_velocity,float2* gpu_star_acceleration,float* gpu_star_weight,float4 position_limits,curandState d_state)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
  		if ( idx>=n_star )
  		{
  		 return;
  		}
  		if (gpu_star_position[idx].x<position_limits.x || gpu_star_position[idx].y<position_limits.y ||gpu_star_position[idx].x>position_limits.z ||gpu_star_position[idx].y>position_limits.w)
  		{
  			gpu_star_position[idx].x=position_limits.x+curand_uniform(d_state)*(position_limits.z-position_limits.x);
  			gpu_star_position[idx].y=position_limits.y+curand_uniform(d_state)*(position_limits.w-position_limits.y);
  			gpu_star_velocity[idx].x=0;
  			gpu_star_velocity[idx].y=0;
  			gpu_star_acceleration[idx].x=0;
  			gpu_star_acceleration[idx].y=0;
  			gpu_star_weight[idx]=0;
  		}
	}
	__global__ void initialization_kernel(int n_star,float2* gpu_star_position,float2* gpu_star_velocity,float2* gpu_star_acceleration,float* gpu_star_weight,float4 position_limits,curandState d_state)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
  		if ( idx>=n_star )
  		{
  		 return;
  		}
  		if (gpu_star_position[idx].x<position_limits.x || gpu_star_position[idx].y<position_limits.y ||gpu_star_position[idx].x>position_limits.z ||gpu_star_position[idx].y>position_limits.w)
  		{
  			gpu_star_position[idx].x=position_limits.x+curand_normal(d_state)*(position_limits.z-position_limits.x);
  			gpu_star_position[idx].y=position_limits.y+curand_normal(d_state)*(position_limits.w-position_limits.y);
  			gpu_star_velocity[idx].x=curand_normal(d_state);
  			gpu_star_velocity[idx].y=curand_normal(d_state);
  			gpu_star_acceleration[idx].x=0;
  			gpu_star_acceleration[idx].y=0;
  			gpu_star_weight[idx]=curand_uniform(d_state)*5;
  		}
	}
	//Constructor
	Stars::Stars(int nStar)
	 : n_star(nStar)
	 {
	 	assert(nStar>0);
	 	cudaMalloc(&gpu_star_position, * n_star*sizeof(float2));
	 	cudaMalloc(&gpu_star_velocity, * n_star*sizeof(float2));
	 	cudaMalloc(&gpu_star_acceleration, * n_star*sizeof(float2));
	 	cudaMalloc(&gpu_star_weight,n_star*sizeof(float));
	 }
	//Destructor
	Stars::~Solver()
	{
		cudaFree(gpu_star_position);
		cudaFree(gpu_star_velocity);
		cudaFree(gpu_star_acceleration);
		cudaFree(gpu_star_weight);
	}

	void Stars::initialize(float4 position_limit)
	{
		curandState *d_state;
  		cudaMalloc(&d_state, sizeof(curandState));
  		setup_kernel<<<1,1>>>(d_state);
  		const int kNumThread_pointwise = 1024;
  		const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  		initialize_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,gpu_star_velocity,gpu_star_acceleration,gpu_star_weight,position_limits,d_state)
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
  		visualize_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,board,gpu_star_weight, size_th,width,height, visualization_limits);	
	}

	void Stars::deletion_check(float4 position_limits)
	{
		curandState *d_state;
  		cudaMalloc(&d_state, sizeof(curandState));
  		setup_kernel<<<1,1>>>(d_state);
  		const int kNumThread_pointwise = 1024;
  		const int kNumBlock_pointwise  = ((n_star-1)/kNumThread_pointwise)+1;
  		deletion_check_kernel<<<kNumBlock_pointwise,kNumThread_pointwise>>>(n_star,gpu_star_position,gpu_star_velocity,gpu_star_acceleration,gpu_star_weight,position_limits,d_state)
	}

#endif