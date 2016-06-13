////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/postdo.cu
/// @brief   Test nbfmm::Solver::postdo
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"
#include <algorithm>
#include <numeric>

using namespace nbfmm;
using namespace std;

void TestNbfmmSolver::postdo() {
  Solver& solver = *ptr_solver;
  cudaError_t cuda_status;

  // Allocate memory
  float2 effect_origin[num_particle];
  int    perm[num_particle];

  // Alias vectors
  auto effect               = random_position;
  auto gpuptr_effect_origin = gpuptr_float2;

  // Create random permutation
  iota(perm, perm+num_particle, 0);
  random_shuffle(perm, perm+num_particle);

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_effect_, effect, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_perm_,   perm,   num_particle * sizeof(int),    cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Run postdo
  solver.postdo(num_particle, gpuptr_effect_origin);

  // Copy output vectors
  cuda_status = cudaMemcpy(effect_origin, gpuptr_effect_origin, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < num_particle; ++i ) {
    CPPUNIT_ASSERT(effect[i] == effect_origin[perm[i]]);
  }
}
