////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    demo/demo.cu
/// @brief   The demo code
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#pragma warning
#define NBFMM_CHECK

#include <cstdio>
#include <iostream>
#include <random>
#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

using namespace std;
using namespace nbfmm;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main function
///
int main() {
  cout << "NBFMM "
       << NBFMM_VERSION_MAJOR << "."
       << NBFMM_VERSION_MINOR << "."
       << NBFMM_VERSION_PATCH << " demo" << endl;

  const int    num_level        = 3;
  const int    max_num_particle = 16;
  const int    num_particle     = 10;
  const float4 position_limits  = make_float4(0, -1, 8, 3);

  Solver solver(num_level, max_num_particle, position_limits);

  float2 *position, *gpuptr_position, *effect, *gpuptr_effect, *effect0;
  float  *weight, *gpuptr_weight;
  position = (float2*) malloc(max_num_particle * sizeof(float2));
  weight   = (float*)  malloc(max_num_particle * sizeof(float));
  effect   = (float2*) malloc(max_num_particle * sizeof(float2));
  effect0  = (float2*) malloc(max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_position, max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_weight,   max_num_particle * sizeof(float));
  cudaMalloc(&gpuptr_effect,   max_num_particle * sizeof(float2));

  // Generate data
  default_random_engine generator;
  uniform_real_distribution<float> position_x_rand(position_limits.x, position_limits.z);
  uniform_real_distribution<float> position_y_rand(position_limits.y, position_limits.w);
  exponential_distribution<float>  weight_rand(1.0);
  for ( auto i = 0; i < num_particle; ++i ) {
    position[i].x = position_x_rand(generator);
    position[i].y = position_y_rand(generator);
    weight[i]     = weight_rand(generator);
  }

  // Compute effects
  for ( auto i = 0; i < num_particle; ++i ) {
    effect0[i] = make_float2(0.0f, 0.0f);
    for ( auto j = 0; j < num_particle; ++j ) {
      if ( i != j ) {
        effect0[i] += kernelFunction(position[i], position[j], weight[j]);
      }
    }
  }

  // Solve by FMM
  cudaMemcpy(gpuptr_position, position, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuptr_weight,   weight,   num_particle * sizeof(float),  cudaMemcpyHostToDevice);
  solver.solve(num_particle, gpuptr_position, gpuptr_weight, gpuptr_effect);
  cudaMemcpy(effect,   gpuptr_effect,   num_particle * sizeof(float2), cudaMemcpyDeviceToHost);

#pragma warning
  int2 *index_sorted = (int2*) malloc(max_num_particle * sizeof(int2));
  int2 *index        = (int2*) malloc(max_num_particle * sizeof(int2));
  int  *perm         = (int*)  malloc(max_num_particle * sizeof(int));
  cudaMemcpy(index_sorted, solver.gpuptr_index_, num_particle * sizeof(int2), cudaMemcpyDeviceToHost);
  cudaMemcpy(perm,         solver.gpuptr_perm_,  num_particle * sizeof(int),  cudaMemcpyDeviceToHost);
  for ( auto i = 0; i < num_particle; ++i ) {
    index[perm[i]] = index_sorted[i];
  }

  // Display data
  printf("    Position\t\t\t    Weight\t    Index\t    Effect(CPU)\t\t\t    Effect(FMM)\n");
  for ( auto i = 0; i < num_particle; ++i ) {
    printf("(%12.8f, %12.8f) \t%12.8f \t(%4d, %4d) \t(%12.8f, %12.8f) \t(%12.8f, %12.8f)\n",
           position[i].x, position[i].y, weight[i], index[i].x, index[i].y,
           effect0[i].x, effect0[i].y, effect[i].x, effect[i].y);
  }
  printf("\n");
  return 0;
}
