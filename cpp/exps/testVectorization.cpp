#include <immintrin.h>
#include <random>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>

#include "./common.h"

// icc -xhost  -O3 /workspace/algorithm/1_cpp/2_avx/gbt_exp/exp5_onedal.cpp
// 

static const int smpLen  = 1024;


void prepareSmp(float* smpArr) {

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<float> urdSmp(0, 1);

    // init sample
    for (int i =0; i < smpLen; ++i){
        smpArr[i] = urdSmp(e);
    }
}

void prepare_idx(unsigned int* index) {
    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<unsigned> rindex(0, smpLen-1);

    for (int i = 0; i < smpLen; i++){
        index[i] = rindex(e);
    }
}

void core(std::vector<float>& a, std::vector<float>& b, unsigned int* idx) {
    
    #pragma ivdep
    #pragma vector always
    for (int j = 0; j < smpLen; ++ j) {
        a[idx[j]] += b[j];
    }
}



int main(int argc, char** argv) {
    // 
    std::vector<float> a(smpLen, 0.f);
    std::vector<float> b(smpLen, 1.f);

    unsigned int* index = new unsigned int[smpLen];
    prepare_idx(index);
    
    // 
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < cycleNum; ++ i) {
        core(a, b, index);
    }
    auto end = std::chrono::system_clock::now(); 

    std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
    std::cout << a[10] << std::endl;

}
