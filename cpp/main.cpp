// #include <immintrin.h>
#include <random>
#include <chrono>
#include <cstdlib>

#include "group_tree.h"



void prepareSmp(float* smpArr) {

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<float> urdSmp(0, 1);

    // init sample
    for (int i =0; i < smpLen; ++i){
        smpArr[i] = urdSmp(e);
    }
}

// 
int main(int argc, char** argv) {
    
    // prepare sample
    float smpArr[100];
    prepareSmp(smpArr);
    
    GBTreeModel gbt1 = GBTreeModel();
    // 
    int cycleNum = 100000;
    float res = 0.;
    // auto start = std::chrono::system_clock::now();
    // for (int i = 0; i < cycleNum; ++ i){
    //     res = gbt1.predictGBT(smpArr);
    // }
    res = gbt1.testTime(smpArr);
    // auto end = std::chrono::system_clock::now(); 

    // std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
    std::cout << "result: " << res << std::endl;

}
