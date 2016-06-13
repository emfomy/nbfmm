////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/m2m.cu
/// @brief   Test nbfmm::Solver::m2m
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"
#include <algorithm>
#include <random>

using namespace std;
using namespace nbfmm;

void TestNbfmmSolver::m2m() {
  Solver& solver = *ptr_solver;
  cudaError_t cuda_status;

  // Allocate memory
  float2 cell_position0[num_level][base_dim][base_dim];
  float2 cell_position[num_level][base_dim][base_dim];
  float  cell_weight0[num_level][base_dim][base_dim];
  float  cell_weight[num_level][base_dim][base_dim];

  // Generate cell position and weight
  default_random_engine generator;
  uniform_real_distribution<float> rand_position_x(position_limits.x, position_limits.z);
  uniform_real_distribution<float> rand_position_y(position_limits.y, position_limits.w);
  exponential_distribution<float>  rand_weight(1.0);
  for ( auto j = 0; j < base_dim; ++j ) {
    for ( auto i = 0; i < base_dim; ++i ) {
      cell_position0[0][j][i].x = rand_position_x(generator);
      cell_position0[0][j][i].y = rand_position_y(generator);
      cell_weight0[0][j][i]     = rand_weight(generator);
    }
  }

  // Compute effects
  for ( auto l = 1; l < num_level; ++l ) {
    int cell_size = 1 << l;
    int shift = cell_size / 2;
    #pragma omp parallel for
    for ( auto j = 0; j < base_dim; j += cell_size ) {
      for ( auto i = 0; i < base_dim; i += cell_size ) {
        cell_position0[l][j][i] = cell_position0[l-1][j][i]             * cell_weight0[l-1][j][i]
                                + cell_position0[l-1][j][i+shift]       * cell_weight0[l-1][j][i+shift]
                                + cell_position0[l-1][j+shift][i]       * cell_weight0[l-1][j+shift][i]
                                + cell_position0[l-1][j+shift][i+shift] * cell_weight0[l-1][j+shift][i+shift];
        cell_weight0[l][j][i]   = cell_weight0[l-1][j][i]
                                + cell_weight0[l-1][j][i+shift]
                                + cell_weight0[l-1][j+shift][i]
                                + cell_weight0[l-1][j+shift][i+shift];
        cell_position0[l][j][i] /= cell_weight0[l][j][i];
      }
    }
  }

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_cell_position_, cell_position0,
                           base_dim * base_dim * sizeof(float2), cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_cell_weight_,   cell_weight0,
                           base_dim * base_dim * sizeof(float),  cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Run p2m
  solver.m2m();

  // Copy output vectors
  cuda_status = cudaMemcpy(cell_position, solver.gpuptr_cell_position_,
                           base_dim * base_dim * num_level * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(cell_weight,   solver.gpuptr_cell_weight_,
                           base_dim * base_dim * num_level * sizeof(float),  cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  for ( auto l = 0; l < num_level; ++l ) {
    int cell_size = 1 << l;
    for ( auto j = 0; j < base_dim; j += cell_size ) {
      for ( auto i = 0; i < base_dim; i += cell_size ) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_position[l][j][i].x, cell_position0[l][j][i].x, 1e-4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_position[l][j][i].y, cell_position0[l][j][i].y, 1e-4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_weight[l][j][i],     cell_weight0[l][j][i],     1e-4);
      }
    }
  }
}
