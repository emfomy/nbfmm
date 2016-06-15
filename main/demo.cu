////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    main/demo.cu
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
       << NBFMM_VERSION_PATCH << " demo" << endl << endl;

  const int    num_level        = 4;
  const int    max_num_particle = 256;
  const int    num_particle     = 250;
  const float4 position_limits  = make_float4(0, -1, 8, 3);

  Solver solver(num_level, max_num_particle, position_limits);

  float2 position[max_num_particle];
  float2 effect0[max_num_particle];
  float2 effect[max_num_particle];
  float  weight[max_num_particle];

  float2 *gpuptr_position, *gpuptr_effect;
  float  *gpuptr_weight;
  cudaMalloc(&gpuptr_position, max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_effect,   max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_weight,   max_num_particle * sizeof(float));

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
  int2 index_sorted[max_num_particle];
  int2 index[max_num_particle];
  int  perm[max_num_particle];
  cudaMemcpy(index_sorted, solver.gpuptr_index_, num_particle * sizeof(int2), cudaMemcpyDeviceToHost);
  cudaMemcpy(perm,         solver.gpuptr_perm_,  num_particle * sizeof(int),  cudaMemcpyDeviceToHost);
  for ( auto i = 0; i < num_particle; ++i ) {
    index[perm[i]] = index_sorted[i];
  }

  // Display data
  printf("    Position\t\t\t    Weight\t    Index\t    Effect(CPU)\t\t\t    Effect(FMM)\n");
  for ( auto i = 0; i < num_particle; ++i ) {
    printf("(%12.4f, %12.4f) \t%12.4f \t(%4d, %4d) \t(%12.4f, %12.4f) \t(%12.4f, %12.4f)\n",
           position[i].x, position[i].y, weight[i], index[i].x, index[i].y,
           effect0[i].x, effect0[i].y, effect[i].x, effect[i].y);
  }
  printf("\n");

// #pragma warning
//   const int base_dim = solver.base_dim_;
//   const int total_num_cell = base_dim * base_dim * num_level;
//   float2 cell_position[num_level][base_dim][base_dim];
//   float2 cell_effect[num_level][base_dim][base_dim];
//   float  cell_weight[num_level][base_dim][base_dim];
//   cudaMemcpy(cell_position, solver.gpuptr_cell_position_, total_num_cell * sizeof(float2), cudaMemcpyDeviceToHost);
//   cudaMemcpy(cell_effect,   solver.gpuptr_cell_effect_,   total_num_cell * sizeof(float2), cudaMemcpyDeviceToHost);
//   cudaMemcpy(cell_weight,   solver.gpuptr_cell_weight_,   total_num_cell * sizeof(float),  cudaMemcpyDeviceToHost);

//   // Display cell data
//   for ( auto l = 0; l < num_level; ++l ) {
//     int cell_size = 1 << l;
//     for ( auto j = 0; j < base_dim; j += cell_size ) {
//       for ( auto i = 0; i < base_dim; i += cell_size ) {
//         printf("#%2d (%4d, %4d): (%12.4f, %12.4f) * %12.4f -> (%12.4f, %12.4f)\n", l, i, j,
//                cell_position[l][j][i].x,  cell_position[l][j][i].y,  cell_weight[l][j][i],
//                cell_effect[l][j][i].x,    cell_effect[l][j][i].y);
//       }
//     }
//   }
//   printf("\n");

  // Free memory
  cudaFree(gpuptr_position);
  cudaFree(gpuptr_effect);
  cudaFree(gpuptr_weight);

  return 0;
}
