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

  /// The maximum number of points
  const int max_num_point_;

  /// The kernel function
  const KernelFunction kernel_function_;

 public:

  /// Default constructor
  Solver(
      const int num_level,                                  ///< the number of grid levels
      const int max_num_point,                              ///< the maximum number of points
      const KernelFunction kernel_function = KernelGravity  ///< the kernel function, default as nbfmm::KernelGravity
  );

  /// Default destructor
  ~Solver();

};

}  // namespace lorasc

#endif  // NBFMM_SOLVER_HPP_
