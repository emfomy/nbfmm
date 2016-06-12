////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/predo.cu
/// @brief   Test nbfmm::Solver::predo
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

using namespace nbfmm;
using namespace std;

void TestNbfmmSolver::predo() {
  Solver& solver = *ptr_solver;
  cudaError_t cuda_status;

  // Allocate memory
  float2 position[num_particle];
  float  weight[num_particle];
  int2   index[num_particle];
  int    perm[num_particle];
  int    head[num_cellp1];

  // Alias vectors
  auto position_origin = random_uniform2;
  auto weight_origin   = random_exponential;
  auto gpuptr_position_origin = gpuptr_float2;
  auto gpuptr_weight_origin   = gpuptr_float;

  // Copy input vectors
  cuda_status = cudaMemcpy(gpuptr_position_origin, position_origin, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(gpuptr_weight_origin,   weight_origin,   num_particle * sizeof(float),  cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Run predo
  solver.predo(num_particle, gpuptr_position_origin, gpuptr_weight_origin);

  // Copy output vectors
  cuda_status = cudaMemcpy(position, solver.gpuptr_position_, num_particle * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(weight,   solver.gpuptr_weight_,   num_particle * sizeof(float),  cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(index,    solver.gpuptr_index_,    num_particle * sizeof(int2),   cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(perm,     solver.gpuptr_perm_,     num_particle * sizeof(int),    cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(head,     solver.gpuptr_head_,     num_cellp1   * sizeof(int),    cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < num_particle; ++i ) {
    CPPUNIT_ASSERT(position[i] == position_origin[perm[i]]);
    CPPUNIT_ASSERT(weight[i]   == weight_origin[perm[i]]);
    CPPUNIT_ASSERT(position[i].x >= index[i].x    * base_cell_size.x + position_limits.x);
    CPPUNIT_ASSERT(position[i].x < (index[i].x+1) * base_cell_size.x + position_limits.x);
    CPPUNIT_ASSERT(position[i].y >= index[i].y    * base_cell_size.y + position_limits.y);
    CPPUNIT_ASSERT(position[i].y < (index[i].y+1) * base_cell_size.y + position_limits.y);
  }
  for ( auto y = 0; y < base_dim; ++y ) {
    for ( auto x = 0; x < base_dim; ++x ) {
      int idx = x + y * base_dim;
      for ( auto i = head[idx]; i < head[idx+1]; ++i ) {
        CPPUNIT_ASSERT(index[i].x == x);
        CPPUNIT_ASSERT(index[i].y == y);
      }
    }
  }
}
