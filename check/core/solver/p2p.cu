////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/p2p.cu
/// @brief   Test nbfmm::Solver::p2p
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"
#include <algorithm>

using namespace nbfmm;
using namespace std;

void TestNbfmmSolver::p2p() {
  Solver& solver = *ptr_solver;
  cudaError_t cuda_status;

  // Alias vectors
  auto position = random_position;
  auto weight   = random_weight;
  auto index    = random_index;
  auto head     = random_head;

  // Allocate memory
  float2 effect0[num_particle];
  float2 effect[num_particle];

  // Compute effects
  #pragma omp parallel for
  for ( auto i = 0; i < num_particle; ++i ) {
    effect0[i] = make_float2(0.0f, 0.0f);
    for ( auto j = 0; j < num_particle; ++j ) {
      if ( abs(index[i].x-index[j].x) <= 1 && abs(index[i].y-index[j].y) <= 1 && i != j ) {
        effect0[i] += kernelFunction(position[i], position[j], weight[j]);
      }
    }
  }

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_position_, position, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_weight_,   weight,   num_particle * sizeof(float),  cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_index_,    index,    num_particle * sizeof(int2),   cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_head_,     head,     num_cell_p1  * sizeof(int),    cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Run p2p
  solver.p2p(num_particle);

  // Copy output vectors
  cuda_status = cudaMemcpy(effect, solver.gpuptr_effect_, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < num_particle; ++i ) {
    CPPUNIT_ASSERT_DOUBLES_EQUAL(effect0[i].x, effect[i].x, 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(effect0[i].y, effect[i].y, 1e-4);
  }
}
