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

  nbfmm::Solver* ptr_solver;

  const int    num_level        = 4;
  const int    base_dim         = 1 << (num_level-1);
  const int    max_num_particle = 64;
  const int    num_particle     = 60;
  const int    num_cell_p1      = base_dim * base_dim + 1;
  const float4 position_limits  = make_float4(0, -1, 8, 2);
  const float2 base_cell_size   = make_float2((position_limits.z - position_limits.x) / base_dim,
                                              (position_limits.w - position_limits.y) / base_dim);

  float2* random_uniform2;
  float*  random_exponential;

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
