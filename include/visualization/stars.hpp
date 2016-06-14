#ifndef DEMO_STARS_HPP
#define DEMO_STARS_HPP
#include <nbfmm/config.hpp>
class Stars 
{
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
}

#endif