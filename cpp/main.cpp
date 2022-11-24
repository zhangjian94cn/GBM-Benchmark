// #include <immintrin.h>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#include "group_tree.h"
#include "frontend.h"
#include "cnpy.h"

#include <ittnotify.h>
#include <omp.h>

#include <tbb/tbb.h>
#include <tbb/task_arena.h>
#include "oneapi/tbb.h"
#include <tbb/task_scheduler_observer.h>

__itt_domain* domain = __itt_domain_create("predict");
__itt_string_handle* handle_main = __itt_string_handle_create("test.zhangjian");


// class pinning_observer : public oneapi::tbb::task_scheduler_observer {
// public:
//     pinning_observer( oneapi::tbb::task_arena &a)
//         : oneapi::tbb::task_scheduler_observer(a){
//         observe(true); // activate the observer
//     }
//     void on_scheduler_entry( bool worker ) override {
//         set_thread_affinity(oneapi::tbb::this_task_arena::current_thread_index(), m_mask);
//     }
//     void on_scheduler_exit( bool worker ) override {
//         restore_thread_affinity();
//     }
// };

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
    const int iblock = 50, iblockT = 100;
    const int n = dataDim / iblock;
    const int nT = 1000 / iblockT;
    
    // exp1 openmp
    // #pragma omp parallel for
    // for (int i = 0; i < dataDim; ++ i) {
    //     res[i] = gbt.predictGBT(data + i * featDim);
    // }

    // exp2 blocked openmp
    // #pragma omp parallel for
    // for (int i = 0; i < n; i++) {
    //     const int offset = i*iblock;
    //     for (int j = 0 ;j < iblock; j++) {
    //         res[offset + j] = gbt.predictGBT(data + offset * featDim );
    //     }
    // }

    // exp3 tbb
    // tbb::parallel_for(0, dataDim, 1, [&](int i) {
    //         res[i] = gbt.predictGBT(data + i * featDim);
    // });

    // set to physical core number
    // oneapi::tbb::task_arena arena(144);
    // pinning_observer po = pinning_observer(arena)

    // arena.execute([&]{
    //     tbb::parallel_for(0, n, 1, [&](int i) {
    //         const int offset = i * iblock;
    //         for (int j = 0 ;j < iblock; j++) {
    //             res[offset + j] = gbt.predictGBT(data + offset * featDim );
    //         }
    //     });
    // });

    // static tbb::affinity_partitioner ap;
    // static tbb::static_partitioner sp;
    // static tbb::simple_partitioner sp;
    // static tbb::auto_partitioner ap;

    // tbb::parallel_for(0, n, 1, [&](int i) {
    //     // GBTreeModel _gbt = gbt;
    //     const int offset = i * iblock;
    //     for (int j = 0 ;j < iblock; j++) {
    //         // res[offset + j] = gbt.predictGBT(data);
    //         res[offset + j] = gbt.predictGBT(data + offset * featDim );
    //     }
    //     // res[offset] = gbt.predictGBT(data + offset * featDim );
    // });
    // // }, ap);

    // tbb::parallel_for(
    //     tbb::blocked_range<size_t>(0, n), 
    //     [&](const tbb::blocked_range<size_t>& r){
        
    //     for(size_t i = r.begin(); i!=r.end(); i++) {
    //         const int offset = i * iblock;
    //         for (int j = 0 ;j < iblock; j++) {
    //             res[offset + j] = gbt.predictGBT(data + offset * featDim );
    //         }
    //     }
    // });

    // tbb::parallel_for(
    //     tbb::blocked_range2d<size_t>(0, nT, 0, n), 
    //     [&](const tbb::blocked_range2d<size_t>& r){
        
    //     for(size_t i = r.cols().begin(); i != r.cols().end(); i++)
    //     for(size_t j = r.rows().begin(); j != r.rows().end(); j++)
    //     {   
    //         // printf("i is %d \n", i);
    //         // printf("j is %d \n", j);
    //         const int offset = i * iblock, offsetT = j * iblockT;
    //         for (int k = 0 ;k < iblock; ++ k) {
    //             for (int kT = 0 ;kT < iblockT; ++ kT) {
    //             // res[(offset + k) * 1000 + j] = gbt.predictGBT(data + offset * featDim);
    //                 res[(offset + k) * 1000 + offsetT + kT] = \
    //                     gbt._trees[offsetT + kT].predictTree(data + offset * featDim);
    //             }
    //         }
    //         // a[i*n+j] = std::sin(i) * std::sin(j);
    //     }
    // });

    // 
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, nT, 0, n), 
        [&](const tbb::blocked_range2d<size_t>& r){
        
        for(size_t i = r.cols().begin(); i != r.cols().end(); i++)
        for(size_t j = r.rows().begin(); j != r.rows().end(); j++)
        {   
            // printf("i is %d \n", i);
            // printf("j is %d \n", j);
            const int offset = i * iblock, offsetT = j * iblockT;
            for (int kT = 0 ;kT < iblockT; ++ kT) {
                for (int k = 0 ;k < iblock; ++ k) {
                // res[(offset + k) * 1000 + j] = gbt.predictGBT(data + offset * featDim);
                    res[(offset + k) * 1000 + offsetT + kT] = \
                        gbt._trees[offsetT + kT].predictTree(data + offset * featDim);
                }
            }
        }
    });



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

    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep8_256.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep8_128.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_4_16.json";
    std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_dep4_8.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep4_8.json";
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
    // std::vector<float> res(featDim * dataDim);
    std::vector<float> res(featDim * dataDim * 1000);
    for (int i = 0; i < featDim * dataDim; ++ i) {
        // smpX[i] = loaded_dataX[i] - 0.384;
        smpX[i] = loaded_dataX[i];
        // smpY[i] = loaded_dataY[i];
    } 
    // printf("here \n");


    auto start = std::chrono::system_clock::now();
    // __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
    for (int k = 0; k < 100; ++ k) {
        pred_core(gbt, smpX.data(), dataDim, featDim, res);
        // pred_core(gbt, smpX.data() + featDim * dataDim / 2, dataDim / 2, featDim, res);
    }
    // pred_core(gbt, smpX.data(), dataDim, featDim, res);
    // __itt_task_end(domain);

    auto end = std::chrono::system_clock::now(); 
    std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
    
    // print result
    for (int i = 0; i < dataDim; ++ i) {
        if (i % 400000 == 0) {
            std::cout << res[i] << std::endl;
        }
    }

}

int main() {

    test2();
    return 0;
}