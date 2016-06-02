////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/solver.hpp
/// @brief   The definition of the FMM solver.
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_SOLVER_HPP_
#define NBFMM_SOLVER_HPP_

#include <nbfmm/config.hpp>
#include <nbfmm/kernel_functions.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM.
//
namespace nbfmm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// The NBFMM solver class.
///
class Solver {

 protected:

  /// The number of grid levels.
  const int num_level_;

  /// The number of girds in the base level per side.
  const int base_size_;

  /// The maximum number of particles
  const int max_num_particle_;

  /// The limits of positions. [x_min, y_min, x_max, y_max].
  const float4 position_limits_;

  /// The kernel function
  const KernelFunction kernel_function_;

  /// The device pointer of sorted particle positions. @n Vector, 1 by @p max_num_particle_.
  float2* gpuptr_position;

  /// The device pointer of sorted particle effects. @n Vector, 1 by @p max_num_particle_.
  float2* gpuptr_effect;

  /// The device pointer of sorted particle weights. @n Vector, 1 by @p max_num_particle_.
  float*  gpuptr_weight;

  /// The device pointer of sorted particle indices. @n Vector, 1 by @p max_num_particle_.
  int*    gpuptr_index;

  /// The device pointer of sorted heads of particle indices in each grid. @n Vector, 1 by @p base_size_^2.
  int*    gpuptr_head;

  /// The pitched pointers of multipole grids.
  cudaPitchedPtr pitchedptr_multipole;

  /// The pitched pointers of local grids.
  cudaPitchedPtr pitchedptr_local;

  /// The device pointer of multipole grids. @n Vector, @p base_size_ by @p base_size_ by @p num_level_.
  float*&  gpuptr_multipole = reinterpret_cast<float*&>(pitchedptr_multipole.ptr);

  /// The device pointer of local grids. @n Vector, @p base_size_ by @p base_size_ by @p num_level_.
  float2*& gpuptr_local     = reinterpret_cast<float2*&>(pitchedptr_local.ptr);

 public:

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Default constructor
  ///
  /// @param  num_level         the number of grid levels.
  /// @param  max_num_particle  the maximum number of particles.
  /// @param  position_limits   the limits of positions. [x_min, y_min, x_max, y_max].
  /// @param  kernel_function   the kernel function, default as nbfmm::kernelGravity.
  ///
  Solver( const int num_level, const int max_num_particle,
          const float4 position_limits, const KernelFunction kernel_function = kernelGravity );

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Default destructor
  ///
  ~Solver();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Solve system
  ///
  /// @param[in]   num_particle            the maximum number of particles.
  /// @param[in]   gpuptr_position_origin  the device pointer of original particle positions.
  /// @param[in]   gpuptr_weight           the device pointer of original particle weights.
  /// @param[out]  gpuptr_effect_origin    the device pointer of original particle effects.
  ///
  void solve( const int num_particle, const float2* gpuptr_position_origin,
              const float* gpuptr_weight_origin, const float2* gpuptr_effect_origin );

 protected:

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Compute particle to particle
  ///
  void p2p();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Compute particle to multipole
  ///
  void p2m();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Compute multipole to multipole
  ///
  void m2m();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Compute multipole to local
  ///
  void m2l();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Compute local to particle
  ///
  void l2p();

};

}  // namespace nbfmm

#endif  // NBFMM_SOLVER_HPP_
