////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver.cu
/// @brief   Test nbfmm::Solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#define NBFMM_TEST_CORE_SOLVER_CU_

#include <cstdio>
#include <random>
#include "solver.hpp"

using namespace std;

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestNbfmmSolver, "Solver");

TestNbfmmSolver::TestNbfmmSolver() {
  ptr_solver = new Solver(num_level, max_num_particle, position_limits);

  position = (float2*) malloc(max_num_particle          * sizeof(float2));
  effect   = (float2*) malloc(max_num_particle          * sizeof(float2));
  weight   = (float*)  malloc(max_num_particle          * sizeof(float));
  index    = (int2*)   malloc(max_num_particle          * sizeof(int2));
  perm     = (int*)    malloc(max_num_particle          * sizeof(int));
  head     = (int*)    malloc((base_dim * base_dim + 1) * sizeof(int));

  cell_position = (float2*) malloc(max_num_particle * sizeof(float2));
  cell_effect   = (float2*) malloc(max_num_particle * sizeof(float2));
  cell_weight   = (float*)  malloc(max_num_particle * sizeof(float));

  position_origin = (float2*) malloc(base_dim * base_dim * num_level * sizeof(float2));
  effect_origin   = (float2*) malloc(base_dim * base_dim * num_level * sizeof(float2));
  weight_origin   = (float*)  malloc(base_dim * base_dim * num_level * sizeof(float));

  cudaMalloc(&gpuptr_position_origin,      max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_effect_origin,        max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_weight_origin,        max_num_particle * sizeof(float));

  default_random_engine generator;
  uniform_real_distribution<float> position_x_rand(position_limits.x, position_limits.z);
  uniform_real_distribution<float> position_y_rand(position_limits.y, position_limits.w);
  exponential_distribution<float>  weight_rand(1.0);
  for ( auto i = 0; i < num_particle; ++i ) {
    position_origin[i].x = position_x_rand(generator);
    position_origin[i].y = position_y_rand(generator);
    weight_origin[i]     = weight_rand(generator);
  }
}

TestNbfmmSolver::~TestNbfmmSolver() {
  delete ptr_solver;

  free(position);
  free(effect);
  free(weight);
  free(index);
  free(perm);
  free(head);

  free(cell_position);
  free(cell_effect);
  free(cell_weight);

  free(position_origin);
  free(effect_origin);
  free(weight_origin);

  cudaFree(gpuptr_position_origin);
  cudaFree(gpuptr_effect_origin);
  cudaFree(gpuptr_weight_origin);
}
