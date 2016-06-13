////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver.cu
/// @brief   Test nbfmm::Solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "solver.hpp"
#include <algorithm>
#include <numeric>
#include <random>

using namespace nbfmm;
using namespace std;

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestNbfmmSolver, "Solver");

const float4 TestNbfmmSolver::position_limits  = make_float4(0, -1, 8, 2);

TestNbfmmSolver::TestNbfmmSolver() : solver(num_level, num_particle, position_limits) {
  // Allocate memory
  cudaMalloc(&gpuptr_float2, num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_float,  num_particle * sizeof(float));

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
  float2 (*cell_position)[base_dim][base_dim] = (float2(*)[base_dim][base_dim]) random_cell_position;
  float  (*cell_weight)[base_dim][base_dim]   = (float (*)[base_dim][base_dim]) random_cell_weight;
  for ( auto l = 0; l < num_level; ++l ) {
    int cell_size = 1 << l;
    for ( auto j = 0; j < base_dim; j += cell_size ) {
      for ( auto i = 0; i < base_dim; i += cell_size ) {
        cell_position[l][j][i].x = rand_position_x(generator);
        cell_position[l][j][i].y = rand_position_y(generator);
        cell_weight[l][j][i]     = rand_weight(generator);
      }
    }
  }

  // Generate head
  random_head[0] = 0;
  int rand_sum = 0;
  for ( auto i = 1; i < num_cell_p1; ++i ) {
    rand_sum += rand_head(generator);
    random_head[i] = rand_sum * num_particle;
  }
  for ( auto i = 0; i < num_cell_p1; ++i ) {
    random_head[i] /= rand_sum;
  }

  // Create random permutation
  iota(random_perm, random_perm+num_particle, 0);
  random_shuffle(random_perm, random_perm+num_particle);

  // Generate index
  for ( auto i = 0; i < base_dim * base_dim; ++i ) {
    fill(random_index+random_head[i], random_index+random_head[i+1], make_int2(i % base_dim, i / base_dim));
  }
}

TestNbfmmSolver::~TestNbfmmSolver() {
  cudaFree(gpuptr_float2);
  cudaFree(gpuptr_float);
}
