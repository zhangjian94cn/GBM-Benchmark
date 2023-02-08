#include "group_tree.h"

float GBTreeModel::testTimePre(float* smpArr) {
    return _trees[0].testTime(smpArr);
    // return _trees[0]->testTime(smpArr);
}

float GBTreeModel::testTime(float* smpArr) {
    float res = 0.f;
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < Common::cycleNum; ++ i){
        res = _trees[0].predictTree(smpArr);
        // res = _trees[0]->predictTree(smpArr);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
    return res;
}


