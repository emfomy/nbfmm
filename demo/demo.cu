////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    demo/demo.cu
/// @brief   The demo code
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <cstdio>
#include <iostream>
#include <random>
#include <nbfmm.hpp>

using namespace std;
using namespace nbfmm;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main function
///
int main( int argc, char *argv[] ) {
  cout << "NBFMM "
       << NBFMM_VERSION_MAJOR << "."
       << NBFMM_VERSION_MINOR << "."
       << NBFMM_VERSION_PATCH << " demo" << endl;

  const int    num_level        = 4;
  const int    max_num_particle = 16;
  const int    num_particle     = 10;
  const float4 position_limits  = make_float4(0, -1, 8, 3);

  Solver solver(num_level, max_num_particle, position_limits);

  float2 *position, *gpuptr_position, *effect, *gpuptr_effect;
  float  *weight, *gpuptr_weight;
  position = (float2*) malloc(max_num_particle * sizeof(float2));
  weight   = (float*)  malloc(max_num_particle * sizeof(float));
  effect   = (float2*) malloc(max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_position, max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_weight,   max_num_particle * sizeof(float));
  cudaMalloc(&gpuptr_effect,   max_num_particle * sizeof(float2));

  default_random_engine generator;
  uniform_real_distribution<float> position_x_rand(position_limits.x, position_limits.z);
  uniform_real_distribution<float> position_y_rand(position_limits.y, position_limits.w);
  exponential_distribution<float>  weight_rand(1.0);
  for ( auto i = 0; i < num_particle; ++i ) {
    position[i].x = position_x_rand(generator);
    position[i].y = position_y_rand(generator);
    weight[i]     = weight_rand(generator);
  }
  for ( auto i = 0; i < num_particle; ++i ) {
    printf("(%12.8f, %12.8f) \t%12.8f\n", position[i].x, position[i].y, weight[i]);
  }
  printf("\n");
  cudaMemcpy(gpuptr_position, position, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuptr_weight,   weight,   num_particle * sizeof(float),  cudaMemcpyHostToDevice);

  solver.solve(num_particle, gpuptr_position, gpuptr_weight, gpuptr_effect);

  cudaMemcpy(position, gpuptr_position, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(weight,   gpuptr_weight,   num_particle * sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(effect,   gpuptr_effect,   num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  for ( auto i = 0; i < num_particle; ++i ) {
    printf("(%12.8f, %12.8f) \t%12.8f \t(%12.8f, %12.8f)\n", position[i].x, position[i].y, weight[i], effect[i].x, effect[i].y);
  }
  return 0;
}
