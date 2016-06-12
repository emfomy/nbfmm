////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver.cu
/// @brief   Test nbfmm::Solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include "solver.hpp"
#include <random>

using namespace nbfmm;
using namespace std;

CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestNbfmmSolver, "Solver");

TestNbfmmSolver::TestNbfmmSolver() {
  ptr_solver = new Solver(num_level, max_num_particle, position_limits);

  random_uniform2    = (float2*) malloc(max_num_particle * sizeof(float2));
  random_exponential = (float*)  malloc(max_num_particle * sizeof(float));

  cudaMalloc(&gpuptr_float2, max_num_particle * sizeof(float2));
  cudaMalloc(&gpuptr_float,  max_num_particle * sizeof(float));

  // Create random vectors
  default_random_engine generator;
  uniform_real_distribution<float> uniform_x_rand(position_limits.x, position_limits.z);
  uniform_real_distribution<float> uniform_y_rand(position_limits.y, position_limits.w);
  exponential_distribution<float>  exponential_rand(1.0);
  for ( auto i = 0; i < num_particle; ++i ) {
    random_uniform2[i].x  = uniform_x_rand(generator);
    random_uniform2[i].y  = uniform_y_rand(generator);
    random_exponential[i] = exponential_rand(generator);
  }
}

TestNbfmmSolver::~TestNbfmmSolver() {
  delete ptr_solver;
  free(random_uniform2);
  free(random_exponential);
  cudaFree(gpuptr_float2);
  cudaFree(gpuptr_float);
}
