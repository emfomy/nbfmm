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

  int num_particle = 2;

  // Alias vectors
  auto position = random_uniform2;
  auto weight   = random_exponential;

  // Allocate memory
  float2 effect0[num_particle];
  float2 effect[num_particle];
  int2   index[num_particle];
  int    head[num_cell_p1];

  // Fill index and head
  int num0 = 0;
  int num1 = num_particle;
  int num2 = num_particle;
  int num3 = num_particle;
  int num4 = num_particle;
  head[0]          = num0;
  head[1]          = num1;
  head[base_dim]   = num2;
  head[base_dim+1] = num3;
  fill(head+2,          head+base_dim,    num2);
  fill(head+base_dim+2, head+num_cell_p1, num4);
  fill(index+num0, index+num1, make_int2(0, 0));
  fill(index+num1, index+num2, make_int2(0, 1));
  fill(index+num2, index+num3, make_int2(1, 0));
  fill(index+num3, index+num4, make_int2(1, 1));

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
  cuda_status = cudaMemcpy(solver.gpuptr_head_,     head,     num_cell_p1  * sizeof(int),    cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Run predo
  solver.p2p(num_particle);

  // Copy output vectors
  cuda_status = cudaMemcpy(effect, solver.gpuptr_effect_, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < num_particle; ++i ) {
    CPPUNIT_ASSERT(abs(effect[i].x - effect0[i].x) < 1e-4);
    CPPUNIT_ASSERT(abs(effect[i].y - effect0[i].y) < 1e-4);
  }
}
