////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/l2l.cu
/// @brief   Test nbfmm::Solver::l2l
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

void TestNbfmmSolver::l2l() {
  cudaError_t cuda_status;

  // Allocate memory
  float2 cell_effect0[num_level][base_dim][base_dim];
  float2 cell_effect[base_dim][base_dim];

  // Copy random vectors
  memcpy(cell_effect0, random_cell_position, base_dim * base_dim * num_level * sizeof(float2));

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_cell_effect_, cell_effect0, base_dim * base_dim * num_level * sizeof(float2),
                           cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Compute effects
  for ( auto l = num_level-1; l > 0; --l ) {
    int cell_size = 1 << l;
    int offset = cell_size / 2;
    for ( auto j = 0; j < base_dim; j += cell_size ) {
      for ( auto i = 0; i < base_dim; i += cell_size ) {
        cell_effect0[l-1][j][i]               += cell_effect0[l][j][i];
        cell_effect0[l-1][j][i+offset]        += cell_effect0[l][j][i];
        cell_effect0[l-1][j+offset][i]        += cell_effect0[l][j][i];
        cell_effect0[l-1][j+offset][i+offset] += cell_effect0[l][j][i];
      }
    }
  }

  // Run l2l
  solver.l2l();

  // Copy output vectors
  cuda_status = cudaMemcpy(cell_effect, solver.gpuptr_cell_effect_, base_dim * base_dim * sizeof(float2),
                           cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto j = 0; j < base_dim; ++j ) {
    for ( auto i = 0; i < base_dim; ++i ) {
      // printf("\n (%2d, %2d): (%12.4f, %12.4f) | (%12.4f, %12.4f)", i, j,
      //        cell_effect0[0][j][i].x, cell_effect0[0][j][i].y, cell_effect[j][i].x, cell_effect[j][i].y);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_effect0[0][j][i].x, cell_effect[j][i].x, 1e-4);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_effect0[0][j][i].y, cell_effect[j][i].y, 1e-4);
    }
  }
}
