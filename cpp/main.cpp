// #include <immintrin.h>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#include "group_tree_acc.h"
#include "frontend.h"
#include "cnpy.h"

#include <ittnotify.h>
#include <omp.h>

#include <tbb/tbb.h>
#include <tbb/task_arena.h>
#include "oneapi/tbb.h"
#include <tbb/task_scheduler_observer.h>

// __itt_domain* domain = __itt_domain_create("predict");
// __itt_string_handle* handle_main = __itt_string_handle_create("test.zhangjian");


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
    for (int i =0; i < smpLen; ++i) {
        smpArr[i] = urdSmp(e);
    }
}

void pred_core(
    GBTreeModel &gbt, 
    float* data, 
    const int dataDim, 
    const int treeDim, 
    const int featDim, 
    std::vector<float> &res,
    std::vector<float> &tmp
    ) {
    
    const int iblockD = 50, iblockT = 1;
    const int nD = dataDim / iblockD;
    const int nT = treeDim / iblockT;
    
    

    // // exp1 openmp
    // // #pragma omp parallel for
    // for (int i = 0; i < dataDim; ++ i) {
    //     res[i] = gbt.predictGBT(data + i * featDim);
    // }   

    // // exp2 blocked openmp
    // #pragma omp parallel for
    // for (int i = 0; i < nD; i++) {
    //     const int offset = i*iblockD;
    //     for (int j = 0 ;j < iblockD; j++) {
    //         res[offset + j] = gbt.predictGBT(data + offset * featDim );
    //     }
    // }

    // // exp3 tbb, RESULT CORRECT
    // tbb::parallel_for(0, dataDim, 1, [&](int i) {
    //         res[i] = gbt.predictGBT(data + i * featDim);
    // });

    // // exp4 set to physical core number
    // oneapi::tbb::task_arena arena(144);
    // pinning_observer po = pinning_observer(arena)

    // arena.execute([&]{
    //     tbb::parallel_for(0, nD, 1, [&](int i) {
    //         const int offset = i * iblockD;
    //         for (int j = 0 ;j < iblockD; j++) {
    //             res[offset + j] = gbt.predictGBT(data + offset * featDim );
    //         }
    //     });
    // });

    // // exp5 memory affinity
    // static tbb::affinity_partitioner ap;
    // static tbb::static_partitioner sp;
    // static tbb::xgboost_partitioner sp;
    // static tbb::auto_partitioner ap;

    // tbb::parallel_for(0, nD, 1, [&](int i) {
    //     // GBTreeModel _gbt = gbt;
    //     const int offset = i * iblockD;
    //     for (int j = 0 ;j < iblockD; j++) {
    //         // res[offset + j] = gbt.predictGBT(data);
    //         res[offset + j] = gbt.predictGBT(data + (offset + j) * featDim );
    //     }
    //     // res[offset] = gbt.predictGBT(data + offset * featDim );
    // });
    // // }, ap);

    // // exp6 1D blocked_range parallel
    // tbb::parallel_for(
    //     tbb::blocked_range<size_t>(0, nD), 
    //     [&](const tbb::blocked_range<size_t>& r){
        
    //     for(size_t i = r.begin(); i!=r.end(); i++) {
    //         const int offset = i * iblockD;
    //         for (int j = 0 ;j < iblockD; j++) {
    //             res[offset + j] = gbt.predictGBT(data + (offset + j) * featDim );
    //         }
    //     }
    // });

    // // exp7 2D block range parallel
    // tbb::parallel_for(
    //     tbb::blocked_range2d<size_t>(0, nT, 0, nD), 
    //     [&](const tbb::blocked_range2d<size_t>& r){
        
    //     for(size_t i = r.cols().begin(); i != r.cols().end(); i++) {
    //         for(size_t j = r.rows().begin(); j != r.rows().end(); j++) {
    //             // printf("i is %d \n", i);
    //             // printf("j is %d \n", j);
    //             const int offsetD = i * iblockD, offsetT = j * iblockT;
    //             for (int kD = 0 ;kD < iblockD; ++ kD) {
    //                 for (int kT = 0 ;kT < iblockT; ++ kT) {
    //                 // res[(offsetD + kD) * treeDim + j] = gbt.predictGBT(data + offsetD * featDim);
    //                     // _res[(offsetD + kD) * treeDim + offsetT + kT] = \
    //                     //     gbt._trees[offsetT + kT].predictTree(data + (offsetD + kD) * featDim);
    //                     res[offsetD + kD] += \
    //                         gbt._trees[offsetT + kT].predictTree(data + (offsetD + kD) * featDim);
    //                 }
    //             }
    //             // a[i*nD+j] = std::sin(i) * std::sin(j);
    //             // res[j] += _res[treeDim * j + i];
    //             // res[i] = sigmoid(res[i]);
    //         }
    //     }
    // });
    

    // exp 8 2D blocked parallel (tree outer, data inner) and use memory affinity
    static tbb::affinity_partitioner ap;
    tbb::parallel_for(
        tbb::blocked_range2d<size_t>(0, nT, 0, nD), 
        [&](const tbb::blocked_range2d<size_t>& r){
        
        for(size_t i = r.cols().begin(); i != r.cols().end(); i++)
        {
            // std::vector<float> tmp(nT);
            for(size_t j = r.rows().begin(); j != r.rows().end(); j++)
            {
                // printf("i is %d \n", i);
                // printf("j is %d \n", j);
                const int offsetD = i * iblockD, offsetT = j * iblockT;
                for (int kT = 0 ;kT < iblockT; ++ kT) {
                    for (int kD = 0 ;kD < iblockD; ++ kD) {
                        // res[(offsetD + kD) * 1000 + j] = gbt.predictGBT(data + offsetD * featDim);
                        // tmp[j] += \
                        //     gbt._treeAggs[offsetT + kT].predict(data + (offsetD + kD) * featDim);
                        tmp[(offsetD + kD) * nT + j] += \
                            gbt._treeAggs[offsetT + kT].predict(data + (offsetD + kD) * featDim);
                        // res[offsetD + kD] += \
                        //     gbt._treeAggs[offsetT + kT].predict(data + (offsetD + kD) * featDim);
                    }
                }
                // res[offsetD + kD] += tmp[j];
            }
        }
    // });
    }, ap);

    tbb::parallel_for(0, nD, 1, [&](int i) {
        const int offsetD = i * iblockD;
        for (int j = 0 ;j < iblockD; j++) {
            float dt = 0.f;
            for (int k = 0; k < nT; ++ k) {
                dt += tmp[(offsetD + j) * nT + k];
            }
            res[offsetD + j] = sigmoid(dt);
        }
    });
    // }, ap);


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
    // res = gbt1.testTime(smpArr);
    // auto end = std::chrono::system_clock::now(); 

    // std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
    std::cout << "result: " << res << std::endl;

}


void test2() {

    // std::string modelPath = "/workspace/GBM-Benchmark/test.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/test_10_8_256.json";
    std::string modelPath = "/workspace/GBM-Benchmark/test_full.json";

    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep8_256.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep8_128.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_4_16.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_8_256full.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_8_256.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_10_8_256.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_8_256.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1k_dep4_8.json";
    // std::string modelPath = "/workspace/GBM-Benchmark/xgb-higgs-model-1_6_1-ntrees_1_dep4_8.json";
    std::string dataPathX = "/workspace/GBM-Benchmark/data/higgs_intel/higgs1m_x_test.npy";
    std::string dataPathY = "/workspace/GBM-Benchmark/data/higgs_intel/higgs1m_y_test.npy";
    
    GBTreeModel gbt = GBTreeModel();
    // LoadXGBoostJSONModel(modelPath.c_str(), gbt);
    LoadTreeAggJSONModel(modelPath.c_str(), gbt);

    cnpy::NpyArray arrX = cnpy::npy_load(dataPathX);
    cnpy::NpyArray arrY = cnpy::npy_load(dataPathY);
    float* loaded_dataX = arrX.data<float>();
    float* loaded_dataY = arrY.data<float>();

    int featDim = arrX.shape[1];
    int dataDim = arrX.shape[0];
    int treeDim = gbt.getTreeNum();
    std::vector<float> smpX(featDim * dataDim), smpY(featDim * dataDim);
    std::vector<float> res(dataDim);
    // std::vector<float> res(dataDim * treeDim);
    
    // std::vector<float> res(dataDim);
    for (int i = 0; i < featDim * dataDim; ++ i) {
        smpX[i] = loaded_dataX[i];
        // smpY[i] = loaded_dataY[i];
    } 
    // printf("here \n");

    auto start = std::chrono::system_clock::now();
    // __itt_task_begin(domain, __itt_null, __itt_null, handle_main);
    //for (int k = 0; k < 100; ++ k) {
    const int nT = treeDim / 1;
    std::vector<float> tmp(dataDim * nT);
    for (int k = 0; k < 100; ++ k) {
        pred_core(gbt, smpX.data(), dataDim, treeDim, featDim, res, tmp);
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
            // std::cout << i<<" "<<res[i] << std::endl;
        }
    }

}

int main() {

    test2();
    return 0;
}
