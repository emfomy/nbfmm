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

	//Constructor
	Stars(int nStar);
	//Destructor
	~Stars();
	//initialize
	void initialize();
	//update
	void update(int FPS);
}

#endif