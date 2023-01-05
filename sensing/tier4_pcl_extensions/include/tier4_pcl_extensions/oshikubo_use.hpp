#pragma once

#include <pcl/common/common.h>
#include <pcl/common/point_tests.h> // for isXYZFinite
#include <tier4_pcl_extensions/voxel_grid_covariance.h>

#include <boost/random/mersenne_twister.hpp> // for mt19937
#include <boost/random/normal_distribution.hpp> // for normal_distribution
#include <boost/random/variate_generator.hpp> // for variate_generator
#include <random>
#include <fstream>




int size_d1,size_d2,size_d3;
//std::vector<int> duplication_check_d1,duplication_check_d2,duplication_check_d3;
std::vector<int> vec_num_output_d1,vec_num_output_d2,vec_num_output_d3;

//ある範囲内での整数を一様分布で発生させる関数の定義
        //std::random_device rd; 
        //static std::mt19937 gen(std::random_device rd()); //Mersenne twister algorithm
        //int low,high;
        //std::uniform_int_distribution<> ud_int(int low, int high);
        
        //std::random_device seed_gen;
        //static std::default_random_engine engine(std::random_device seed_gen());
        //float f_low, f_high;
        //std::uniform_real_distribution<float> ud(float f_low, float f_high);

        //std::vector<int> duplication_check;
        
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


