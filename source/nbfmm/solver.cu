////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/solver.cu
/// @brief   The implementation of the FMM solver
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#include <nbfmm/solver.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  The namespace NBFMM
//
namespace nbfmm {

// Default constructor
Solver::Solver(
    const int num_level,
    const int max_num_point,
    const KernelFunction kernel_function
) : num_level_(num_level),
    max_num_point_(max_num_point),
    kernel_function_(kernel_function) {
}

// Default destructor
Solver::~Solver() {
}

}  // namespace lorasc
