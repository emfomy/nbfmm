////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/solver.hpp
/// @brief   The definition of the FMM solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_SOLVER_HPP_
#define NBFMM_SOLVER_HPP_

#include <nbfmm/config.hpp>
#include <nbfmm/kernel_functions.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The NBFMM solver class
///
class Solver {

 protected:

  /// The number of grid levels
  const int num_level_;

  /// The size of base grid
  const int size_base_grid_;

  /// The maximum number of particles
  const int max_num_point_;

  /// The kernel function
  const KernelFunction kernel_function_;

  /// The device pointer of particle positions
  float2* gpuptr_position;

  /// The device pointer of particle effects
  float2* gpuptr_effect;

  /// The device pointer of particle indices
  int*    gpuptr_index;

  /// The device pointer of heads of particle indices in each grid
  int*    gpuptr_head;

  /// The device pointer of multipole grids
  float*  gpuptr_multipole;

  /// The device pointer of local grids
  float2* gpuptr_local;

  /// The pitched pointers of multipole grids
  cudaPitchedPtr pitchedptr_multipole;

  /// The pitched pointers of local grids
  cudaPitchedPtr pitchedptr_local;

 public:

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Default constructor
  ///
  /// @param  num_level        the number of grid levels
  /// @param  max_num_point    the maximum number of particles
  /// @param  kernel_function  the kernel function, default as nbfmm::KernelGravity
  ///
  Solver( const int num_level, const int max_num_point, const KernelFunction kernel_function = KernelGravity );

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Default destructor
  ///
  ~Solver();

};

}  // namespace nbfmm

#endif  // NBFMM_SOLVER_HPP_
