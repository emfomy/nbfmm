////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/visualization/stars.hpp
/// @brief   The definition of stars class
///
/// @author  Mu Yang <emfomy@gmail.com>
/// @author  Yung-Kang Lee <blasteg@gmail.com>
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
	const int n_star;
	//position of stars
	float2* gpu_star_position;
	//velocity of stars
	float2* gpu_star_velocity;
	//acceleration of stars
	float2* gpu_star_acceleration;
	//weight of stars
	float* gpu_star_weight;

	//Constructor
	Stars(int nStar);
	//Destructor
	~Stars();
	//initialize
	void initialize(float4 position_limit);
	//update
	void update(int FPS);

	void visualize(int width, int height, uint8_t *board,float size_th,float4 visualization_limits);

	void deletion_check(float4 position_limits);
};

#endif  // DEMO_STARS_VISUALIZATION_HPP_

/// @endcond