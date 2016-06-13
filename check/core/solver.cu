////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver.cu
/// @brief   Test nbfmm::Solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "solver.hpp"
#include <algorithm>
#include <random>

using namespace nbfmm;
using namespace std;

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestNbfmmSolver, "Solver");

TestNbfmmSolver::TestNbfmmSolver() {
  ptr_solver = new Solver(num_level, max_num_particle, position_limits);

  // Allocate memory
  random_position = (float2*) malloc(max_num_particle * sizeof(float2));
  random_weight   = (float*)  malloc(max_num_particle * sizeof(float));
  random_index    = (int2*)   malloc(max_num_particle * sizeof(int2));
  random_head     = (int*)    malloc(num_cell_p1 * sizeof(int));
  cudaMalloc(&gpuptr_float2, max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_float,  max_num_particle * sizeof(float));

  // Create random generator
  default_random_engine generator;
  uniform_real_distribution<float> rand_position_x(position_limits.x, position_limits.z);
  uniform_real_distribution<float> rand_position_y(position_limits.y, position_limits.w);
  exponential_distribution<float>  rand_weight(1.0);
  uniform_int_distribution<int>    rand_head(0, 255);

  // Generate position and weight
  for ( auto i = 0; i < num_particle; ++i ) {
    random_position[i].x = rand_position_x(generator);
    random_position[i].y = rand_position_y(generator);
    random_weight[i]     = rand_weight(generator);
  }

  // Generate head
  random_head[0] = 0;
  int rand_sum = 0;
  for ( auto i = 1; i < num_cell_p1; ++i ) {
    rand_sum += rand_head(generator);
    random_head[i] = rand_sum * num_particle;
  }
  #pragma omp parallel for
  for ( auto i = 0; i < num_cell_p1; ++i ) {
    random_head[i] /= rand_sum;
  }

  // Generate index
  #pragma omp parallel for
  for ( auto i = 0; i < base_dim * base_dim; ++i ) {
    fill(random_index+random_head[i], random_index+random_head[i+1], make_int2(i % base_dim, i / base_dim));
  }
}

TestNbfmmSolver::~TestNbfmmSolver() {
  delete ptr_solver;
  free(random_position);
  free(random_weight);
  free(random_index);
  free(random_head);
  cudaFree(gpuptr_float2);
  cudaFree(gpuptr_float);
}
