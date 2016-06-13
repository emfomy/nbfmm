////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/p2m.cu
/// @brief   Test nbfmm::Solver::p2m
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

using namespace nbfmm;
using namespace std;

void TestNbfmmSolver::p2m() {
  cudaError_t cuda_status;

  // Alias vectors
  auto position = random_position;
  auto weight   = random_weight;
  auto index    = random_index;
  auto head     = random_head;

  // Allocate memory
  float2 cell_position0[base_dim * base_dim];
  float2 cell_position[base_dim * base_dim];
  float  cell_weight0[base_dim * base_dim];
  float  cell_weight[base_dim * base_dim];

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_position_, position, num_particle * sizeof(float2), cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_weight_,   weight,   num_particle * sizeof(float),  cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_index_,    index,    num_particle * sizeof(int2),   cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_head_,     head,     num_cell_p1  * sizeof(int),    cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Compute cell positions and weights
  for ( auto i = 0; i < base_dim * base_dim; ++i ) {
    cell_position0[i] = make_float2(0, 0);
    cell_weight0[i]   = 0;
    for ( auto idx = head[i]; idx < head[i+1]; ++idx ) {
      cell_position0[i] += position[idx] * weight[idx];
      cell_weight0[i]   += weight[idx];
    }
    if ( head[i] != head[i+1] ) {
      cell_position0[i] /= cell_weight0[i];
    }
  }

  // Run p2m
  solver.p2m(num_particle);

  // Copy output vectors
  cuda_status = cudaMemcpy(cell_position, solver.gpuptr_cell_position_,
                           base_dim * base_dim * sizeof(float2), cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(cell_weight,   solver.gpuptr_cell_weight_,
                           base_dim * base_dim * sizeof(float),  cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto i = 0; i < base_dim * base_dim; ++i ) {
    // printf("\n (%d, %d): (%12.4e, %12.4e) * %12.4e | (%12.4e, %12.4e) * %12.4e", i % base_dim, i / base_dim,
    //        cell_position0[i].x, cell_position0[i].y, cell_weight0[i], cell_position[i].x, cell_position[i].y, cell_weight[i]);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_position0[i].x, cell_position[i].x, 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_position0[i].y, cell_position[i].y, 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_weight0[i],     cell_weight[i],     1e-4);
  }
}
