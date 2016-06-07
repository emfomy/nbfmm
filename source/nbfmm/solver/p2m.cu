////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver/p2m.cu
/// @brief   Compute particle to multipole
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

//  The namespace NBFMM
namespace nbfmm {


// P2M
void Solver::p2m(int num_particle) {
	const dim3 kNumThread_gridwise = (32,32,1);
  const dim3 kNumBlock_gridwise  = (((base_size_-1)/kNumThread_gridwise)+1,((base_size_-1)/kNumThread_gridwise)+1,1);
  p2m_kernel<<<kNumBlock_gridwise,kNumThread_gridwise>>>(base_size_,num_particle,gpuptr_position_,gpuptr_weight_,)
  
  /// @todo Implement!
}

}  // namespace nbfmm
