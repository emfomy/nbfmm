#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <nbfmm/visualization/SyncedMemory.h>
#include <nbfmm/visualization/stars.hpp>
#include <nbfmm/core.hpp>
#include <nbfmm/utility.hpp>

int main()
{
	int width=1000;
	int height=700;
	int FPS=60;
	unsigned n_frame=1800;

	int n_star=500;
	float4 visualization_limit=make_float4(0.0,0.0,15.0,15.0);
	float4 position_limit=make_float4(0.0,0.0,15.0,15.0);

	Stars asteroids(n_star); 
	asteroids.initialize(position_limit);

  const int    num_level        = 1;
  const int    max_num_particle = 20000;

  nbfmm::Solver solver(num_level, max_num_particle, position_limit);

	unsigned FRAME_SIZE = width*height*3/2;
	MemoryBuffer<uint8_t> frameb(FRAME_SIZE);
	auto frames = frameb.CreateSync(FRAME_SIZE);
	FILE *fp = fopen("result.y4m", "wb");
	fprintf(fp, "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 C420\n", width, height, FPS, 1);

	for (unsigned j = 0; j < n_frame; ++j) {
		fputs("FRAME\n", fp);
		asteroids.visualize(width, height,frames.get_gpu_wo(),1,visualization_limit);
		fwrite(frames.get_cpu_ro(), sizeof(uint8_t), FRAME_SIZE, fp);

		solver.solve(n_star, asteroids.gpu_star_position, asteroids.gpu_star_weight, asteroids.gpu_star_acceleration);
		asteroids.update(FPS);
		asteroids.deletion_check(position_limit);
	}

	fclose(fp);
	return 0;
}