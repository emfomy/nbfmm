////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    check/core/solver.hpp
/// @brief   Definition of nbfmm::Solver tester
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_CHECK_CORE_SOLVER_HPP_
#define NBFMM_CHECK_CORE_SOLVER_HPP_

#define NBFMM_CHECK

#include <cstdio>
#include <cppunit/extensions/HelperMacros.h>
#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

class TestNbfmmSolver : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestNbfmmSolver);

  CPPUNIT_TEST(predo);
  CPPUNIT_TEST(postdo);
  CPPUNIT_TEST(p2p);
  CPPUNIT_TEST(p2m);
  CPPUNIT_TEST(m2m);
  CPPUNIT_TEST(m2l);
  CPPUNIT_TEST(l2l);
  CPPUNIT_TEST(l2p);

  CPPUNIT_TEST_SUITE_END();

 private:

  nbfmm::Solver solver;

  static const int    num_level        = 3;
  static const int    base_dim         = 1 << (num_level+1);
  static const int    max_num_particle = 256;
  static const int    num_particle     = 250;
  static const int    num_cell_p1      = base_dim * base_dim + 1;
  static const float4 position_limits;

  float2  random_position[num_particle];
  float   random_weight[num_particle];
  int2    random_index[num_particle];
  int     random_perm[num_particle];
  int     random_head[num_cell_p1];

  float2  random_cell_position[num_level][base_dim][base_dim];
  float   random_cell_weight[num_level][base_dim][base_dim];

  float2* gpuptr_float2;
  float*  gpuptr_float;

 public:

  TestNbfmmSolver();
  ~TestNbfmmSolver();

  void predo();
  void postdo();
  void p2p();
  void p2m();
  void m2m();
  void m2l();
  void l2l();
  void l2p();
};

#endif  // NBFMM_CHECK_CORE_SOLVER_HPP_
