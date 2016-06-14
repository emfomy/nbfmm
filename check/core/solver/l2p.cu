////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/l2p.cu
/// @brief   Test nbfmm::Solver::l2p
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

void TestNbfmmSolver::l2p() {
  cudaError_t cuda_status;

  // Alias vectors
  auto index       = random_index;
  auto cell_effect = random_cell_position;

  // Allocate memory
  float2 effect0[num_particle];
  float2 effect[num_particle];

  // Copy random vectors
  memcpy(effect0, random_position, base_dim * base_dim * sizeof(float2));

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_effect_, effect0, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_index_,  index,   num_particle * sizeof(int2),   cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_cell_effect_, cell_effect, base_dim * base_dim * sizeof(float2),
                           cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Compute effects
  for ( auto i = 0; i < num_particle; ++i ) {
    effect0[i] += cell_effect[0][index[i].y][index[i].x];
  }

  // Run l2p
  solver.l2p(num_particle);

  // Copy output vectors
  cuda_status = cudaMemcpy(effect, solver.gpuptr_effect_, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < num_particle; ++i ) {
    // printf("\n #%3d (%2d, %2d): (%12.4f, %12.4f) | (%12.4f, %12.4f)", i, index[i].x, index[i].y,
    //        effect0[i].x, effect0[i].y, effect[i].x, effect[i].y);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(effect0[i].x, effect[i].x, 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(effect0[i].y, effect[i].y, 1e-4);
  }
}
