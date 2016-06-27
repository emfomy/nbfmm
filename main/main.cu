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

  if ( argc <= 13 ) {
    cout << "Usage: " << argv[0] <<
            " <output file> <width> <height> <fps> <time (second)>"
            " <FMM level> <number of stars> <position width> <position height> <display scale> <grav_const> <size_scale>"
            " <model> [model parameters] ..." << endl << endl;
    abort();
  }

  const char* result_y4m       = argv[1];

  const int   width            = atoi(argv[2]);
  const int   height           = atoi(argv[3]);
  const int   fps              = atoi(argv[4]);
  const int   num_second       = atoi(argv[5]);
  const int   num_frame        = num_second * fps;

  const int   fmm_level        = atoi(argv[6]);
  const int   num_star         = atoi(argv[7]);
  const float pos_width        = atof(argv[8]);
  const float pos_height       = atof(argv[9]);
  const float display_scale    = atof(argv[10]);
  const float tick             = 0.05/fps;
  const float grav_const       = atof(argv[11]);
  const float size_scale       = atof(argv[12]);


  const float4 position_limits = make_float4(-pos_width/2, -pos_height/2, pos_width/2, pos_height/2);
  const float4 display_limits  = make_float4(position_limits.x * display_scale, position_limits.y * display_scale,
                                             position_limits.z * display_scale, position_limits.w * display_scale);
  const float2 center_position = make_float2(0.0f, 0.0f);

  const char* model            = argv[13];

  printf("%d x %d pixel, %d fps, %d sec / %d frame\n", width, height, fps, num_second, num_frame);
  printf("%d FMM level, %d stars, tick = %.6f, grav const = %.2f, scale size = %.2f\n",
         fmm_level, num_star, tick, grav_const, size_scale);
  printf("position size = %.2f x %.2f, display scale = %.2f\n",
         pos_width, pos_height, display_scale);
  printf("position range = [%.2f, %.2f] x [%.2f, %.2f], display range = [%.2f, %.2f] x [%.2f, %.2f]\n",
         position_limits.x, position_limits.z, position_limits.y, position_limits.w,
         display_limits.x,  display_limits.z,  display_limits.y,  display_limits.w);
  printf("\n");

  Stars stars(fmm_level, num_star, width, height, fps, tick, grav_const, size_scale, position_limits, display_limits);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if ( !strcmp(model, "circle") ) {
    if ( argc <= 16 ) {
      cout << "Usage: " << argv[0] << " ... circle <radius> <weight> <center weight>" << endl << endl;
      abort();
    }
    const float radius        = atof(argv[14]);
    const float weight        = atof(argv[15]);
    const float weight_center = atof(argv[16]);

    printf("Model Circle: radius = %.2f, weight = %.2f, center weight = %.2f\n\n", radius, weight, weight_center);

    stars.initialize(model::generateCircleCenter,
      num_star, center_position, radius, weight, weight_center
    );
  } else if ( !strcmp(model, "circle-uniform") ) {
    if ( argc <= 16 ) {
      cout << "Usage: " << argv[0] << " ... circle-uniform <radius> <weight> <center weight>" << endl << endl;
      abort();
    }
    const float radius        = atof(argv[14]);
    const float weight        = atof(argv[15]);
    const float weight_center = atof(argv[16]);

    printf("Model Uniform Circle: radius = %.2f, weight = %.2f, center weight = %.2f\n\n", radius, weight, weight_center);

    stars.initialize(model::generateCircleUniformCenter,
      num_star, center_position, radius, weight, weight_center
    );
  } else if ( !strcmp(model, "disk") ) {
    if ( argc <= 15 ) {
      cout << "Usage: " << argv[0] << " ... disk <radius> <weight>" << endl << endl;
      abort();
    }
    const float radius = atof(argv[14]);
    const float weight = atof(argv[15]);

    printf("Model Disk: radius = %.2f, weight = %.2f\n\n", radius, weight);

    stars.initialize(model::generateDisk,
      num_star, center_position, radius, weight
    );
  } else if ( !strcmp(model, "disk-center") ) {
    if ( argc <= 16 ) {
      cout << "Usage: " << argv[0] << " ... disk-center <radius> <weight> <center weight>" << endl << endl;
      abort();
    }
    const float radius        = atof(argv[14]);
    const float weight        = atof(argv[15]);
    const float weight_center = atof(argv[16]);

    printf("Model Centered Disk: radius = %.2f, weight = %.2f, center weight = %.2f\n\n", radius, weight, weight_center);

    stars.initialize(model::generateDiskCenter,
      num_star, center_position, radius, weight, weight_center
    );
  } else if ( !strcmp(model, "disk-static") ) {
    if ( argc <= 15 ) {
      cout << "Usage: " << argv[0] << " ... disk-static <radius> <weight>" << endl << endl;
      abort();
    }
    const float radius        = atof(argv[14]);
    const float weight        = atof(argv[15]);

    printf("Model Centered Disk: radius = %.2f, weight = %.2f\n\n", radius, weight);

    stars.initialize(model::generateDiskStatic,
      num_star, center_position, radius, weight
    );
  } else if ( !strcmp(model, "double-disk") ) {
    if ( argc <= 19 ) {
      cout << "Usage: " << argv[0] <<
              " ... double-disk <proportion 1> <proportion 2> <radius 1> <radius 2> <weight> <eccentricity>" << endl << endl;
      abort();
    }
    const float n1           = atof(argv[14]);
    const float n2           = atof(argv[15]);
    const float p1           = n1 / (n1+n2);
    const float p2           = n2 / (n1+n2);
    const float radius1      = atof(argv[16]);
    const float radius2      = atof(argv[17]);
    const float weight       = atof(argv[18]);
    const float eccentricity = atof(argv[19]);

    assert(n1 >= 0 && n2 >= 0);

    const float2 center_position1 = (make_float2(display_limits.x, display_limits.y) * (1*p1+0*p2) +
                                     make_float2(display_limits.z, display_limits.w) * (1*p1+2*p2)) / 2;
    const float2 center_position2 = (make_float2(display_limits.z, display_limits.w) * (1*p2+0*p1) +
                                     make_float2(display_limits.x, display_limits.y) * (1*p2+2*p1)) / 2;

    printf("Model Double Disk: proportion = %.2f / %.2f, radius = %.2f / %.2f, weight = %.2f, eccentricity = %.2f\n\n",
           n1, n2, radius1, radius2, weight, eccentricity);

    stars.initialize(model::generateDoubleDisk,
      num_star * p1, num_star * p2, center_position1, center_position2, radius1, radius2, weight, eccentricity
    );
  } else if ( !strcmp(model, "double-disk-center") ) {
    if ( argc <= 20 ) {
      cout << "Usage: " << argv[0] <<
              " ... double-disk-center <proportion 1> <proportion 2> <radius 1> <radius 2>"
              " <weight> <center weight> <eccentricity>" << endl << endl;
      abort();
    }
    const float n1             = atof(argv[14]);
    const float n2             = atof(argv[15]);
    const float p1             = n1 / (n1+n2);
    const float p2             = n2 / (n1+n2);
    const float radius1        = atof(argv[16]);
    const float radius2        = atof(argv[17]);
    const float weight         = atof(argv[18]);
    const float weight_center1 = atof(argv[19]);
    const float weight_center2 = atof(argv[20]);
    const float eccentricity   = atof(argv[21]);

    assert(n1 >= 0 && n2 >= 0);

    const float2 center_position1 = (make_float2(display_limits.x, display_limits.y) * (1*p1+0*p2) +
                                     make_float2(display_limits.z, display_limits.w) * (1*p1+2*p2)) / 2;
    const float2 center_position2 = (make_float2(display_limits.z, display_limits.w) * (1*p2+0*p1) +
                                     make_float2(display_limits.x, display_limits.y) * (1*p2+2*p1)) / 2;

    printf("Model Double Disk: proportion = %.2f / %.2f, radius = %.2f / %.2f, weight = %.2f, eccentricity = %.2f\n\n",
           n1, n2, radius1, radius2, weight, eccentricity);

    stars.initialize(model::generateDoubleDiskCenter,
      num_star * p1, num_star * p2, center_position1, center_position2, radius1, radius2,
      weight, weight_center1, weight_center1, eccentricity
    );
  } else if ( !strcmp(model, "rectangle") ) {
    if ( argc <= 16 ) {
      cout << "Usage: " << argv[0] <<
              " ... rectangle <rectangle width> <rectangle height> <maximum weight>" << endl << endl;
      abort();
    }
    const float model_width  = atof(argv[14]);
    const float model_height = atof(argv[15]);
    const float max_weight   = atof(argv[16]);

    printf("Model Rectangle: size = %.2f x %.2f, max weight = %.2f\n\n", model_width, model_height, max_weight);

    stars.initialize(model::generateRectangle,
      num_star, center_position, model_width, model_height, max_weight
    );
  } else {
    cout << "Unknown model " << model << "." << endl;
    abort();
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int progress = 0;
  for ( auto i = 0; i < 100; ++i ) {
    putchar('=');
  }
  putchar('\r'); fflush(stdout);

  int FRAME_SIZE = width*height*3/2;
  MemoryBuffer<uint8_t> frameb(FRAME_SIZE);
  auto frames = frameb.CreateSync(FRAME_SIZE);
  FILE *fp = fopen(result_y4m, "wb");
  fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420\n", width, height, fps, 1);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  fputs("FRAME\n", fp);
  stars.display(frames.get_gpu_wo());
  fwrite(frames.get_cpu_ro(), sizeof(uint8_t), FRAME_SIZE, fp);

  for ( auto j = 1; j < num_frame; ++j) {
    stars.update();
    stars.display(frames.get_gpu_wo());
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
