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
#include <nbfmm/display.hpp>
#include <nbfmm/model.hpp>
#include <nbfmm/utility.hpp>
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

  const char* result_y4m = ( argc > 1 ) ? argv[1] : "nbfmm.y4m";

  const int width      = 1024;
  const int height     = 768;
  const int fps        = 60;
  const int num_second = 10;
  const int num_frame  = num_second * fps;

  const int   num_star    = 10000;
  const int   fmm_level   = 4;
  const float tick        = 0.05/fps;
  const float grav_const  = 1.0f;
  const float size_scale  = 1.0f;

  const float pos_width     = 16.0f;
  const float pos_height    = 12.0f;
  const float display_scale = 0.5;
  const float4 position_limits = make_float4(-pos_width/2, -pos_height/2, pos_width/2, pos_height/2);
  const float4 display_limits  = make_float4(position_limits.x * display_scale,
                                             position_limits.y * display_scale,
                                             position_limits.z * display_scale,
                                             position_limits.w * display_scale);

  int progress = 0;
  for ( auto i = 0; i < 100; ++i ) {
    putchar('=');
  }
  putchar('\r'); fflush(stdout);

  Stars asteroids(fmm_level, num_star, width, height, fps, tick, grav_const, size_scale, position_limits, display_limits);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // const float2 center_position = (make_float2(display_limits.x, display_limits.y) +
  //                                 make_float2(display_limits.z, display_limits.w)) / 2;
  // const float model_width  = (display_limits.z - display_limits.x);
  // const float model_height = (display_limits.w - display_limits.y);

  // asteroids.initialize(model::generateRectangle,
  //     num_star, center_position, model_width, model_height, 36.0f
  // );

  // const float2 center_position = (make_float2(display_limits.x, display_limits.y) +
  //                                 make_float2(display_limits.z, display_limits.w)) / 2;
  // const float radius = (display_limits.w - display_limits.y)/8;

  // asteroids.initialize(model::generateDisk,
  //     num_star, center_position, radius, 1.0f
  // );

  const int n1 = 5;
  const int n2 = 3;
  const float p1 = float(n1) / (n1+n2);
  const float p2 = float(n2) / (n1+n2);
  const float2 center_position1 = (make_float2(display_limits.x, display_limits.y) * (1*p1+0*p2) +
                                   make_float2(display_limits.z, display_limits.w) * (1*p1+2*p2)) / 2;
  const float2 center_position2 = (make_float2(display_limits.z, display_limits.w) * (1*p2+0*p1) +
                                   make_float2(display_limits.x, display_limits.y) * (1*p2+2*p1)) / 2;
  const float radius = (display_limits.w - display_limits.y)/8;
  const float eccentricity = 0.8f;

  asteroids.initialize(model::generateDoubleDisk,
      num_star*p1, num_star*p2, center_position1, center_position2, radius*p1, radius*p2, 1.0f, eccentricity
  );

  // asteroids.initialize(model::generateDoubleDiskCenter,
  //     num_star*p1, num_star*p2, center_position1, center_position2, radius*p1, radius*p2, 1.0f,
  //     num_star*p1, num_star*p1, eccentricity
  // );

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int FRAME_SIZE = width*height*3/2;
  MemoryBuffer<uint8_t> frameb(FRAME_SIZE);
  auto frames = frameb.CreateSync(FRAME_SIZE);
  FILE *fp = fopen(result_y4m, "wb");
  fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420\n", width, height, fps, 1);

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
