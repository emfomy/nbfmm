////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/core/solver/p2p.cu
/// @brief   Compute particle to particle
///
/// @author  Mu Yang <emfomy@gmail.com>
///          Da-Wei Chang <davidzan830@gmail.com>
///

#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

/// The block dimension used in P2P
static const int block_dim_p2p = 64;

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
  float2       total_effect  = make_float2(0.0f, 0.0f);
  const int2   parent_idx    = index[idx];
  const float2 self_position = position[idx];

  int   start_idx, end_idx, cur_idx;

  __shared__ int    work_list[3][2];
  __shared__ float2 temp_effect[block_dim_p2p];

  // Set working list
  if ( rank == 0 ) {
    int par_idx_left  = (parent_idx.x-1 > 0)        ? (parent_idx.x-1) : 0;
    int par_idx_right = (parent_idx.x+1 < base_dim) ? (parent_idx.x+1) : (base_dim-1);

    #pragma unroll
    for( int j = 0; j < 3; ++j ) {
      int par_idx_y = parent_idx.y+(j-1);
      if( par_idx_y >= 0 && par_idx_y < base_dim ) {
        work_list[j][0] = head[par_idx_left  + par_idx_y * base_dim];
        work_list[j][1] = head[par_idx_right + par_idx_y * base_dim + 1];
      } else {
        work_list[j][0] = 0;
        work_list[j][1] = 0;
      }
    }

  }
  __syncthreads();

  // Go through all work list
  #pragma unroll
  for( int work = 0; work < 3; ++work ) {
    start_idx = work_list[work][0];
    end_idx   = work_list[work][1];
    cur_idx   = start_idx + rank;

    const int loop_times = (end_idx - start_idx) / block_dim_p2p + 1;

    // Go through all particles on work list
    for( int i = 0; i < loop_times; ++i ) {

      // Put data into shared memory, without its self 
      if( cur_idx < end_idx && cur_idx != idx ) {
        temp_effect[rank] = nbfmm::kernelFunction(self_position, position[cur_idx], weight[cur_idx]);
      } else {
        temp_effect[rank] = make_float2(0.0f, 0.0f);
      }
      __syncthreads();

      // Reduction
      for( int end = block_dim_p2p/2; end >= 1; end /= 2 ) {
        if( rank < end ) {
          temp_effect[rank] += temp_effect[rank + end];
        }
        __syncthreads();
      }

      total_effect += temp_effect[0];
      __syncthreads();
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
  if ( num_particle > kMaxGridDim ) {
    NaiveP2P<<<grid_dim, block_dim>>>(num_particle, base_dim_, gpuptr_position_,
                                      gpuptr_weight_, gpuptr_index_, gpuptr_head_, gpuptr_effect_);
  } else {
    BlockP2P<<<num_particle, block_dim_p2p>>>(num_particle, base_dim_, gpuptr_position_,
                                              gpuptr_weight_, gpuptr_index_, gpuptr_head_, gpuptr_effect_);
  }
}

}  // namespace nbfmm
