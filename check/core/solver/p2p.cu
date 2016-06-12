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
  auto position = random_uniform2;
  auto weight   = random_exponential;

  // Allocate memory
  float2 effect0[num_particle];
  float2 effect[num_particle];
  int2   index[num_particle];
  int    head[num_cellp1];

  // Fill index and head
  head[0] = 0;
  head[1] = num_particle * .25;
  head[2] = num_particle * .5;
  head[3] = num_particle * .75;
  head[4] = num_particle;
  fill(index+head[0], index+head[1], make_int2(0, 0));
  fill(index+head[1], index+head[2], make_int2(0, 1));
  fill(index+head[2], index+head[3], make_int2(1, 0));
  fill(index+head[3], index+head[4], make_int2(1, 1));

  // Compute effects
  #pragma omp for
  for ( auto i = 0; i < num_particle; ++i ) {
    effect0[i] = make_float2(0.0f, 0.0f);
    for ( auto j = 0; j < num_particle; ++j ) {
      if ( i != j ) {
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
  cuda_status = cudaMemcpy(solver.gpuptr_head_,     head,     num_cellp1   * sizeof(int),    cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Run predo
  solver.p2p(num_particle);

  // Copy output vectors
  cuda_status = cudaMemcpy(effect, solver.gpuptr_effect_, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < num_particle; ++i ) {
    CPPUNIT_ASSERT(effect[i] == effect0[i]);
  }
}
