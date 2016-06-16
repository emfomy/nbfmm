////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/visualization/stars.hpp
/// @brief   The definition of stars class
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
/// @author  Da-Wei Chang <davidzan830@gmail.com>
///

/// @cond

#ifndef DEMO_STARS_VISUALIZATION_HPP_
#define DEMO_STARS_VISUALIZATION_HPP_
#include <nbfmm/config.hpp>
#include <stdlib.h>
#include <cstdint>

class Stars {
 public:
  //number of stars
	int n_star;
	//current position of stars
	float2* gpu_star_position_cur;
	//previous position of stars
	float2* gpu_star_position_pre;
	//acceleration of stars
	float2* gpu_star_acceleration;
	//weight of stars
	float* gpu_star_weight;

	const int FPS;

	const float dt;

	//Constructor
	Stars(int nStar, int FPS);
	//Destructor
	~Stars();
	//initialize
	void initialize(float4 position_limit);
	//update
	void update();

	void visualize(int width, int height, uint8_t *board,float size_th,float4 visualization_limits);

	void deletion_check(float4 position_limits);
};

#endif  // DEMO_STARS_VISUALIZATION_HPP_

/// @endcond
