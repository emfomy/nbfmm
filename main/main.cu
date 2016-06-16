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
#include <nbfmm/visualization.hpp>
#include <SyncedMemory.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main function
///
int main( int argc, char const *argv[] ) {
  cout << "NBFMM "
       << NBFMM_VERSION_MAJOR << "."
       << NBFMM_VERSION_MINOR << "."
       << NBFMM_VERSION_PATCH << endl << endl;

  const char* result_y4m = ( argc > 1 ) ? argv[1] : "nbfmm_result.mp4";

  const int width        = 1024;
  const int height       = 768;
  const int FPS          = 60;
  const unsigned n_frame = 3000;
  const int n_star       = 500;

  float4 visualization_limit = make_float4(1.0f, 1.0f, 7.0f, 7.0f);
  float4 position_limit      = make_float4(0.0f, 0.0f, 8.0f, 8.0f);

  Stars asteroids(n_star);
  asteroids.initialize(position_limit, FPS);

  const int num_level        = 4;
  const int max_num_particle = 20000;

  nbfmm::Solver solver(num_level, max_num_particle, position_limit);

  unsigned FRAME_SIZE = width*height*3/2;
  MemoryBuffer<uint8_t> frameb(FRAME_SIZE);
  auto frames = frameb.CreateSync(FRAME_SIZE);
  FILE *fp = fopen(result_y4m, "wb");
  fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420\n", width, height, FPS, 1);

  for (unsigned j = 0; j < n_frame; ++j) {
    fputs("FRAME\n", fp);
    asteroids.visualize(width, height,frames.get_gpu_wo(),1,visualization_limit);
    fwrite(frames.get_cpu_ro(), sizeof(uint8_t), FRAME_SIZE, fp);

    solver.solve(n_star, asteroids.gpu_star_position_cur, asteroids.gpu_star_weight, asteroids.gpu_star_acceleration);
    asteroids.update(FPS);
    asteroids.deletion_check(position_limit);
  }

  fclose(fp);

  return 0;
}
