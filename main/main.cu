////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    main/main.cu
/// @brief   The main code
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
///

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>
#include <nbfmm/display.hpp>
#include <SyncedMemory.h>

using namespace std;
using namespace nbfmm;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main function
///
int main( int argc, char const *argv[] ) {
  cout << "NBFMM "
       << NBFMM_VERSION_MAJOR << "."
       << NBFMM_VERSION_MINOR << "."
       << NBFMM_VERSION_PATCH << endl << endl;

  const char* result_y4m = ( argc > 1 ) ? argv[1] : "nbfmm_result.mp4";

  const int width     = 1024;
  const int height    = 768;
  const int FPS       = 60;
  const int num_frame = 300;

  const int   num_star    = 1000;
  const int   fmm_level   = 4;
  const float grav_const  = 10.0f;
  const float tick        = 0.1/FPS;
  const float size_scale  = 1.0f;

  const float pos_width     = 16.0f;
  const float pos_height    = 12.0f;
  const float display_scale = 0.8;
  const float4 position_limits = make_float4(-pos_width/2, -pos_height/2, pos_width/2, pos_height/2);
  const float4 display_limits  = make_float4(position_limits.x * display_scale,
                                             position_limits.y * display_scale,
                                             position_limits.z * display_scale,
                                             position_limits.w * display_scale);

  Stars asteroids(fmm_level, num_star, width, height, FPS, grav_const, tick, size_scale, position_limits, display_limits);

  int progress = 0;
  for ( auto i = 0; i < 100; ++i ) {
    putchar('=');
  }
  putchar('\r'); fflush(stdout);

  asteroids.initialize();

  int FRAME_SIZE = width*height*3/2;
  MemoryBuffer<uint8_t> frameb(FRAME_SIZE);
  auto frames = frameb.CreateSync(FRAME_SIZE);
  FILE *fp = fopen(result_y4m, "wb");
  fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420\n", width, height, FPS, 1);

  fputs("FRAME\n", fp);
  asteroids.display(frames.get_gpu_wo());
  fwrite(frames.get_cpu_ro(), sizeof(uint8_t), FRAME_SIZE, fp);

  for ( auto j = 1; j < num_frame; ++j) {
    asteroids.update();
    asteroids.display(frames.get_gpu_wo());
    fputs("FRAME\n", fp);
    fwrite(frames.get_cpu_ro(), sizeof(uint8_t), FRAME_SIZE, fp);

    if ( 100 * j > progress * num_frame ) {
      ++progress;
      putchar('>'); fflush(stdout);
    }
  }
  putchar('\n'); putchar('\n');

  fclose(fp);

  return 0;
}
