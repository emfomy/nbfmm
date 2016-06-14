////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/m2m.cu
/// @brief   Test nbfmm::Solver::m2m
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

void TestNbfmmSolver::m2m() {
  cudaError_t cuda_status;

  // Allocate memory
  float2 cell_position0[num_level][base_dim][base_dim];
  float2 cell_position[num_level][base_dim][base_dim];
  float  cell_weight0[num_level][base_dim][base_dim];
  float  cell_weight[num_level][base_dim][base_dim];

  // Copy random vectors
  memcpy(cell_position0, random_cell_position, base_dim * base_dim * sizeof(float2));
  memcpy(cell_weight0,   random_cell_weight,   base_dim * base_dim * sizeof(float));

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_cell_position_, cell_position0, base_dim * base_dim * sizeof(float2),
                           cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_cell_weight_,   cell_weight0,   base_dim * base_dim * sizeof(float),
                           cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Compute effects
  for ( auto l = 1; l < num_level; ++l ) {
    int cell_size = 1 << l;
    int shift = cell_size / 2;
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

  // Run m2m
  solver.m2m();

  // Copy output vectors
  cuda_status = cudaMemcpy(cell_position, solver.gpuptr_cell_position_, base_dim * base_dim * num_level * sizeof(float2),
                           cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(cell_weight,   solver.gpuptr_cell_weight_,   base_dim * base_dim * num_level * sizeof(float),
                           cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto l = 0; l < num_level; ++l ) {
    int cell_size = 1 << l;
    for ( auto j = 0; j < base_dim; j += cell_size ) {
      for ( auto i = 0; i < base_dim; i += cell_size ) {
        // printf("\n #%d (%2d, %2d): (%12.4f, %12.4f) * %12.4f | (%12.4f, %12.4f) * %12.4f", l, i, j,
        //        cell_position0[l][j][i].x, cell_position0[l][j][i].y, cell_weight0[l][j][i],
        //        cell_position[l][j][i].x,  cell_position[l][j][i].y,  cell_weight[l][j][i],
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_position0[l][j][i].x, cell_position[l][j][i].x, 1e-4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_position0[l][j][i].y, cell_position[l][j][i].y, 1e-4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_weight0[l][j][i],     cell_weight[l][j][i],     1e-4);
      }
    }
  }
}
