////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/p2p.cu
/// @brief   Compute particle to particle
///
/// @author  Mu Yang <emfomy@gmail.com>
///          Da-Wei Chang <davidzan830@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute particle to particle
///
/// @param[in]   num_particle    the number of particles.
/// @param[in]   cell_side_size  the number of girds in the base level per side.
/// @param[in]   position        the particle positions.
/// @param[in]   weight          the particle weights.
/// @param[in]   index           the particle cell indices.
/// @param[in]   head            the starting permutation indices of each cell.
/// @param[out]  effect          the particle effects.
///
__global__
void NaiveP2P(
  const int     num_particle,
  const int     cell_side_size,
  const float2* position,
  const float*  weight,
  const int2*   index,
  const int*    head,
  float2*       effect
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float2 total_effect = make_float2(0.0f, 0.0f);

  if(idx < num_particle) {
    int2 par_idx = index[idx];

    float2 self_position = position[idx];

    // Go through each surrounding cell
    for(int i = -1; i <= 1; ++i) {
      for(int j = -1; j <= 1; ++j) {

        // Check whether this cell exists
        if(par_idx.x + i <  cell_side_size &&
           par_idx.x + i >= 0              &&
           par_idx.y + j <  cell_side_size &&
           par_idx.y + j >= 0) {

          // Go through each particle in this cell
          int cell_idx  = par_idx.x + par_idx.y * cell_side_size;
          int start_idx = head[cell_idx];
          int end_idx   = head[cell_idx + 1];
          for(int k = start_idx; k < end_idx; ++k) {
            // Cannot calculate action to self
            if(k != idx) {
              total_effect += nbfmm::kernelFunction(self_position, position[k], weight[k]);
            }
          }

        }
      }
    }
    effect[idx] = total_effect;
  }

}

//  The namespace NBFMM
namespace nbfmm {

// P2P
void Solver::p2p( const int num_particle ) {
  const int block_size = 512;
  const int num_block = num_particle/block_size + 1;

  NaiveP2P<<<num_block, block_size>>>(num_particle, base_size_,
    gpuptr_position_, gpuptr_weight_, gpuptr_index_, gpuptr_head_, gpuptr_effect_);
}

}  // namespace nbfmm
