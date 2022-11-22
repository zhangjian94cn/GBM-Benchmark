// #include <immintrin.h>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#include "group_tree.h"
#include "frontend.h"
#include "cnpy.h"

#include <ittnotify.h>


__itt_domain* domain = __itt_domain_create("predict");
__itt_string_handle* handle_main = __itt_string_handle_create("test.zhangjian");

void prepareSmp(float* smpArr) {

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<float> urdSmp(0, 1);

    // init sample
    for (int i =0; i < smpLen; ++i){
        smpArr[i] = urdSmp(e);
    }
}

void pred_core(GBTreeModel &gbt, float* data, int dataDim, int featDim, std::vector<float> &res) {
    for (int i = 0; i < dataDim; ++ i) {
        res[i] = gbt.predictGBT(data + i * featDim);
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
    int dataDim = arrX.shape[0];
    std::vector<float> smpX(featDim * dataDim), smpY(featDim * dataDim);
    std::vector<float> res(featDim * dataDim);
    // for (int i = 0; i < featDim * dataDim; ++ i) {
    //     // smpX[i] = loaded_dataX[i] - 0.384;
    //     smpX[i] = loaded_dataX[i];
    //     smpY[i] = loaded_dataY[i];
    // } 
    printf("here \n");



    auto start = std::chrono::system_clock::now();
    __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
    for (int k = 0; k < 1000; ++ k) {
       pred_core(gbt, smpX.data(), dataDim, featDim, res);
    }
    __itt_task_end(domain);

    auto end = std::chrono::system_clock::now(); 
    std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
    
    // print result
    for (int i = 0; i < dataDim; ++ i) {
        if (i % 1000 == 0) {
            std::cout << res[i] << std::endl;
        }
    }

}

int main() {

    test2();
    return 0;
}