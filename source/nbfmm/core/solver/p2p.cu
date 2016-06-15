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
/// @param[in]   num_particle  the number of particles.
/// @param[in]   base_dim      the number of cells in the base level per side.
/// @param[in]   position      the particle positions.
/// @param[in]   weight        the particle weights.
/// @param[in]   index         the particle cell indices.
/// @param[in]   head          the starting permutation indices of each cell.
/// @param[out]  effect        the particle effects.
///
__global__
void NaiveP2P(
  const int     num_particle,
  const int     base_dim,
  const float2* position,
  const float*  weight,
  const int2*   index,
  const int*    head,
  float2*       effect
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if ( idx >= num_particle ) {
    return;
  }
  const int2   parent_idx    = index[idx];
  const float2 self_position = position[idx];
  float2       total_effect  = make_float2(0.0f, 0.0f);

  // Go through neighbor cells
  for ( int j = parent_idx.y-1; j <= parent_idx.y+1; ++j ) {
    for ( int i = parent_idx.x-1; i <= parent_idx.x+1; ++i ) {

      // Check whether this cell exists
      if( i >= 0 && i < base_dim && j >= 0 && j < base_dim ) {

        // Go through particles in this cell
        int cell_idx  = i + j * base_dim;
        int start_idx = head[cell_idx];
        int end_idx   = head[cell_idx+1];
        for ( int k = start_idx; k < end_idx; ++k ) {
          // Cannot calculate action to self
          if ( k != idx ) {
            total_effect += nbfmm::kernelFunction(self_position, position[k], weight[k]);
          }
        }
      }
    }
  }
  effect[idx] = total_effect;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute particle to particle
///
/// @param[in]   num_particle  the number of particles.
/// @param[in]   base_dim      the number of cells in the base level per side.
/// @param[in]   position      the particle positions.
/// @param[in]   weight        the particle weights.
/// @param[in]   index         the particle cell indices.
/// @param[in]   head          the starting permutation indices of each cell.
/// @param[out]  effect        the particle effects.
///
static const int num_block = 64;
__global__
void BlockP2P(
  const int     num_particle,
  const int     base_dim,
  const float2* position,
  const float*  weight,
  const int2*   index,
  const int*    head,
  float2*       effect
) {
  const int    rank          = threadIdx.x;
  const int    idx           = blockIdx.x;
  const int2   parent_idx;
  const float2 self_position;
  float2       total_effect;
  
  if ( rank == 0 ) {
    parent_idx    = index[idx];
    self_position = position[idx];
    total_effect  = make_float2(0.0f, 0.0f);
  }

  // Go through neighbor cells
  for ( int j = parent_idx.y-1; j <= parent_idx.y+1; ++j ) {
    for ( int i = parent_idx.x-1; i <= parent_idx.x+1; ++i ) {

      // Check whether this cell exists
      if( i >= 0 && i < base_dim && j >= 0 && j < base_dim ) {

        // Go through particles in this cell
        int cell_idx  = i + j * base_dim;
        int start_idx = head[cell_idx];
        int end_idx   = head[cell_idx+1];
        for ( int k = start_idx; k < end_idx; ++k ) {
          // Cannot calculate action to self
          if ( k != idx ) {
            total_effect += nbfmm::kernelFunction(self_position, position[k], weight[k]);
          }
        }
      }
    }
  }

  if ( rank == 0 ) {
    effect[idx] = total_effect;
  }

}

//  The namespace NBFMM
namespace nbfmm {

// P2P
void Solver::p2p( const int num_particle ) {
  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_particle-1)/block_dim)+1;

  NaiveP2P<<<grid_dim, block_dim>>>(num_particle, base_dim_,
                                    gpuptr_position_, gpuptr_weight_, gpuptr_index_, gpuptr_head_, gpuptr_effect_);
}

}  // namespace nbfmm
