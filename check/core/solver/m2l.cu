////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver/m2l.cu
/// @brief   Test nbfmm::Solver::m2l
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "../solver.hpp"

void TestNbfmmSolver::m2l() {
  cudaError_t cuda_status;

  // Alias vectors
  auto cell_position = random_cell_position;
  auto cell_weight   = random_cell_weight;

  // Allocate memory
  float2 cell_effect0[num_level][base_dim][base_dim];
  float2 cell_effect[num_level][base_dim][base_dim];

  // Copy random vectors
  memcpy(cell_position, random_cell_position, base_dim * base_dim * sizeof(float2));
  memcpy(cell_weight,   random_cell_weight,   base_dim * base_dim * sizeof(float));

  // Copy input vectors
  cuda_status = cudaMemcpy(solver.gpuptr_cell_position_, cell_position, base_dim * base_dim * num_level * sizeof(float2),
                           cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(solver.gpuptr_cell_weight_,   cell_weight,   base_dim * base_dim * num_level * sizeof(float),
                           cudaMemcpyHostToDevice);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Compute effects
  for ( auto l = 0; l < num_level; ++l ) {
    int cell_size = 1 << l;
    int parent_size = cell_size * 2;
    for ( auto y = 0; y < base_dim; y += cell_size ) { auto py = (y / (2*cell_size)) * (2*cell_size);
      for ( auto x = 0; x < base_dim; x += cell_size ) { auto px = (x / (2*cell_size)) * (2*cell_size);
        cell_effect0[l][y][x] = make_float2(0.0f, 0.0f);
        for ( auto j = 0; j < base_dim; j += cell_size ) { auto pj = (j / (2*cell_size)) * (2*cell_size);
          for ( auto i = 0; i < base_dim; i += cell_size ) { auto pi = (i / (2*cell_size)) * (2*cell_size);
            if ( abs(pi-px) <= parent_size && abs(pj-py) <= parent_size && (abs(i-x) > cell_size || abs(j-y) > cell_size) ) {
              cell_effect0[l][y][x] += nbfmm::kernelFunction(cell_position[l][y][x],
                                                             cell_position[l][j][i], cell_weight[l][j][i]);
            }
          }
        }
      }
    }
  }

  // Run m2l
  solver.m2l();

  // Copy output vectors
  cuda_status = cudaMemcpy(cell_effect, solver.gpuptr_cell_effect_, base_dim * base_dim * num_level * sizeof(float2),
                           cudaMemcpyDeviceToHost);
  CPPUNIT_ASSERT(cuda_status == cudaSuccess);

  // Check
  for ( auto l = 0; l < num_level; ++l ) {
    int cell_size = 1 << l;
    for ( auto j = 0; j < base_dim; j += cell_size ) {
      for ( auto i = 0; i < base_dim; i += cell_size ) {
        // printf("\n #%d (%2d, %2d): (%12.4f, %12.4f) * %12.4f -> (%12.4f, %12.4f) | (%12.4f, %12.4f)", l, i, j,
        //        cell_position[l][j][i].x, cell_position[l][j][i].y, cell_weight[l][j][i],
        //        cell_effect0[l][j][i].x, cell_effect0[l][j][i].y, cell_effect[l][j][i].x, cell_effect[l][j][i].y);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_effect0[l][j][i].x, cell_effect[l][j][i].x, 1e-4);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cell_effect0[l][j][i].y, cell_effect[l][j][i].y, 1e-4);
      }
    }
  }
}
