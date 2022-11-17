// #include <immintrin.h>
#include <random>
#include <chrono>
#include <cstdlib>

#include "group_tree.h"
#include "frontend.h"
#include "cnpy.h"

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
void test1() {
    
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


void test2() {

    std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep4_8.json";
    std::string dataPathX = "/workspace/GBM-Benchmark/data/higgs_intel/higgs1m_x_test.npy";
    std::string dataPathY = "/workspace/GBM-Benchmark/data/higgs_intel/higgs1m_y_test.npy";
    
    GBTreeModel gbt = GBTreeModel();
    LoadXGBoostJSONModel(modelPath.c_str(), gbt);

    cnpy::NpyArray arrX = cnpy::npy_load(dataPathX);
    cnpy::NpyArray arrY = cnpy::npy_load(dataPathY);
    float* loaded_dataX = arrX.data<float>();
    float* loaded_dataY = arrY.data<float>();

    int featDim = arrX.shape[1];
    std::vector<float> smpX(featDim), smpY(featDim);
    for (int i = 0; i < featDim; ++ i) {
        smpX[i] = loaded_dataX[i];
        smpY[i] = loaded_dataY[i];
    } 

    float res = gbt.predictGBT(smpX.data());

}

int main() {

    test2();
    return 0;
}