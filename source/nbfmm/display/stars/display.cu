////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    source/nbfmm/display/stars/display.cu
/// @brief   Update stars' data
///
/// @author  Mu Yang       <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang  <davidzan830@gmail.com>
///

#include <nbfmm/display/stars.hpp>
#include <nbfmm/utility.hpp>

/// @addtogroup impl_display
/// @{

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Generate circle shape particles
///
/// @param[in]   num_star        the number of stars.
/// @param[in]   width           the frame width.
/// @param[in]   height          the frame height.
/// @param[in]   display_limits  the limits of display positions. [x_min, y_min, x_max, y_max].
/// @param[in]   position        the particle positions.
/// @param[in]   weight          the particle weights.
/// @param[out]  board           the frame board
///

__global__ void displayDevice(
  const int     num_star,
  const int     width,
  const int     height,
  const float   size_scale,
  const float4  display_limits,
  const float2* position,
  const float*  weight,
  uint8_t*      board
) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if ( idx >= num_star ) {
    return;
  }

  const float unit_width  = (display_limits.z - display_limits.x) / width;
  const float unit_height = (display_limits.w - display_limits.y) / height;
  const int x    = floor((position[idx].x - display_limits.x) / unit_width);
  const int y    = floor((display_limits.w - position[idx].y) / unit_height);
  const int size = floor(weight[idx] / size_scale) + 1;

  board += x + y * width;

  if ( 0 <= x && x < width && 0 <= y && y <= height ) {
    if ( size >= 1 ) {
      board[ 0 + 0 * width] = 255;
    }
    if ( size >= 2 ) {
      if ( x >= 1 ) {
        board[-1 + 0 * width] = 255;
      }
      if ( x < width-1 ) {
        board[ 1 + 0 * width] = 255;
      }
      if ( y >= 0 ) {
        board[ 0 - 1 * width] = 255;
      }
      if ( y < height-1 ) {
        board[ 0 + 1 * width] = 255;
      }
    }
    if ( size >= 3 ) {
      if ( x >= 1 && y >= 1 ) {
        board[-1 - 1 * width] = 255;
      }
      if ( x >= 1 && y < height-1 ) {
        board[-1 + 1 * width] = 255;
      }
      if ( x < width-1 && y >= 1 ) {
        board[ 1 - 1 * width] = 255;
      }
      if ( x < width-1 && y < height-1 ) {
        board[ 1 + 1 * width] = 255;
      }
    }
    if ( size >= 4 ) {
      if ( x >= 2 ) {
        board[-2 + 0 * width] = 255;
      }
      if ( x < width-2 ) {
        board[ 2 + 0 * width] = 255;
      }
      if ( y >= 0 ) {
        board[ 0 - 2 * width] = 255;
      }
      if ( y < height-2 ) {
        board[ 0 + 2 * width] = 255;
      }
    }
    if ( size >= 5 ) {
      if ( x >= 2 && y >= 1 ) {
        board[-2 - 1 * width] = 255;
      }
      if ( x >= 1 && y >= 2 ) {
        board[-1 - 2 * width] = 255;
      }
      if ( x >= 2 && y < height-2 ) {
        board[-2 + 1 * width] = 255;
      }
      if ( x >= 1 && y < height-2 ) {
        board[-1 + 2 * width] = 255;
      }
      if ( x < width-2 && y >= 1 ) {
        board[ 2 - 1 * width] = 255;
      }
      if ( x < width-2 && y >= 2 ) {
        board[ 1 - 2 * width] = 255;
      }
      if ( x < width-2 && y < height-2 ) {
        board[ 2 + 1 * width] = 255;
      }
      if ( x < width-2 && y < height-2 ) {
        board[ 1 + 2 * width] = 255;
      }
    }
    if ( size >= 6 ) {
      if ( x >= 3 ) {
        board[-3 + 0 * width] = 255;
      }
      if ( x < width-3 ) {
        board[ 3 + 0 * width] = 255;
      }
      if ( y >= 0 ) {
        board[ 0 - 3 * width] = 255;
      }
      if ( y < height-3 ) {
        board[ 0 + 3 * width] = 255;
      }
      if ( x >= 2 && y >= 2 ) {
        board[-2 - 2 * width] = 255;
      }
      if ( x >= 2 && y < height-2 ) {
        board[-2 + 2 * width] = 255;
      }
      if ( x < width-2 && y >= 2 ) {
        board[ 2 - 2 * width] = 255;
      }
      if ( x < width-2 && y < height-2 ) {
        board[ 2 + 2 * width] = 255;
      }
    }
    if ( size >= 7 ) {
      if ( x >= 3 && y >= 1 ) {
        board[-3 - 1 * width] = 255;
      }
      if ( x >= 1 && y >= 3 ) {
        board[-1 - 3 * width] = 255;
      }
      if ( x >= 3 && y < height-3 ) {
        board[-3 + 1 * width] = 255;
      }
      if ( x >= 1 && y < height-3 ) {
        board[-1 + 3 * width] = 255;
      }
      if ( x < width-3 && y >= 1 ) {
        board[ 3 - 1 * width] = 255;
      }
      if ( x < width-3 && y >= 3 ) {
        board[ 1 - 3 * width] = 255;
      }
      if ( x < width-3 && y < height-3 ) {
        board[ 3 + 1 * width] = 255;
      }
      if ( x < width-3 && y < height-3 ) {
        board[ 1 + 3 * width] = 255;
      }
    }
    if ( size >= 8 ) {
      if ( x >= 4 ) {
        board[-4 + 0 * width] = 255;
      }
      if ( x < width-4 ) {
        board[ 4 + 0 * width] = 255;
      }
      if ( y >= 0 ) {
        board[ 0 - 4 * width] = 255;
      }
      if ( y < height-4 ) {
        board[ 0 + 4 * width] = 255;
      }
      if ( x >= 3 && y >= 2 ) {
        board[-3 - 2 * width] = 255;
      }
      if ( x >= 2 && y >= 3 ) {
        board[-2 - 3 * width] = 255;
      }
      if ( x >= 3 && y < height-3 ) {
        board[-3 + 2 * width] = 255;
      }
      if ( x >= 2 && y < height-3 ) {
        board[-2 + 3 * width] = 255;
      }
      if ( x < width-3 && y >= 2 ) {
        board[ 3 - 2 * width] = 255;
      }
      if ( x < width-3 && y >= 3 ) {
        board[ 2 - 3 * width] = 255;
      }
      if ( x < width-3 && y < height-3 ) {
        board[ 3 + 2 * width] = 255;
      }
      if ( x < width-3 && y < height-3 ) {
        board[ 2 + 3 * width] = 255;
      }
    }
  }
}

/// @}

// Display stars
void nbfmm::Stars::display( uint8_t* board ) {
  cudaMemset(board, 0, width_ * height_);
  cudaMemset(board + width_ * height_, 128, width_ * height_/2);

  const int block_dim = kMaxBlockDim;
  const int grid_dim  = ((num_star_-1)/block_dim)+1;
  displayDevice<<<grid_dim,block_dim>>>(num_star_, width_, height_, size_scale_, display_limits_,
                                        gpuptr_position_cur_, gpuptr_weight_, board);

  checkDeletion();
}
