////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/p2p.cu
/// @brief   Compute particle to particle
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

__global__
void test(
  const float2* position,
  const float*  weight,
  float2*       effect
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  effect[idx] = nbfmm::kernelFunction(position[idx], weight[idx]);
}

//  The namespace NBFMM
namespace nbfmm {

// P2P
void Solver::p2p( const int num_perticle ) {
  test<<<1, num_perticle>>>(gpuptr_position_, gpuptr_weight_, gpuptr_effect_);
  /// @todo Implement!
}

}  // namespace nbfmm
