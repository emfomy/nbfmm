////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/predo.cu
/// @brief   Test nbfmm::Solver::predo
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

using namespace nbfmm;

void TestNbfmmSolver::predo() {
  Solver& solver = *ptr_solver;

  cudaMemcpy(gpuptr_position_origin, position_origin, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuptr_weight_origin,   weight_origin,   num_particle * sizeof(float),  cudaMemcpyHostToDevice);

  // Run predo
  solver.predo(num_particle, gpuptr_position_origin, gpuptr_weight_origin);

  // Copy vector
  cudaMemcpy(position, solver.gpuptr_position_, num_particle              * sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(weight,   solver.gpuptr_weight_,   num_particle              * sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(index,    solver.gpuptr_index_,    num_particle              * sizeof(int2),   cudaMemcpyDeviceToHost);
  cudaMemcpy(perm,     solver.gpuptr_perm_,     num_particle              * sizeof(int),    cudaMemcpyDeviceToHost);
  cudaMemcpy(head,     solver.gpuptr_head_,     (base_dim * base_dim + 1) * sizeof(int),    cudaMemcpyDeviceToHost);

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
        printf("\nx=%d, y=%d, idx=%d, index.x=%d, index.y=%d, i=%d", x, y, idx, index[i].x, index[i].y, i);
        CPPUNIT_ASSERT(index[i].x == x);
        CPPUNIT_ASSERT(index[i].y == y);
      }
    }
  }
}
