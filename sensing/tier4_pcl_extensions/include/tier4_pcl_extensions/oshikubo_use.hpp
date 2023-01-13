#pragma once

#include <pcl/common/common.h>
#include <pcl/common/point_tests.h> // for isXYZFinite
#include <tier4_pcl_extensions/voxel_grid_covariance.h>

#include <boost/random/mersenne_twister.hpp> // for mt19937
#include <boost/random/normal_distribution.hpp> // for normal_distribution
#include <boost/random/variate_generator.hpp> // for variate_generator
#include <random>
#include <fstream>
        
int random_int(int low,int high);//{return int_random;}
float random_float(float f_low,float f_high);//{return float_random;}

int random_int(int low, int high){
    //int int_random;
    std::uniform_int_distribution<> ud_int(low,high);
    std::random_device rd;
    std::mt19937 gen(rd());
	int int_random=ud_int(gen);
    return int_random;
}

float random_float(float f_low, float f_high){
    //float float_random;
    std::uniform_real_distribution<float> ud(f_low, f_high);
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
	float float_random=ud(engine);
    return float_random;
}


