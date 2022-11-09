
#include <limits>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <immintrin.h>
#include <random>
#include <chrono>
#include <iostream>
// 

#include "common/common.h"


static const int gnodeNum = sizeof(__m512) / sizeof(int);
static const int smpLen  = 16;
using SampleGroupT = __m512;


union FeatIdx {
    __m512i i;
    int32_t ii[gnodeNum];
};

union FeatVal {
    __m512  v;
    float vv[gnodeNum];
};

union nodeStat {
    __mmask16  m;
    char mm[sizeof(__mmask16) / sizeof(char)];
};

// 
class NodeGroup {

public:
    NodeGroup() {
        _children.resize(gnodeNum, 0);
    }
    NodeGroup(FeatIdx& _fia, FeatVal& _fiv, std::vector<int32_t>& _children) : 
            _fidxArr(_fia), _fvalArr(_fiv), _children(_children) {}

    void initGroupData(int featNum, int nodeNum) {
        std::random_device rd;
        std::default_random_engine e(rd());
        std::uniform_real_distribution<float> urdSmp(0, 1);

        // init tree 
        std::uniform_int_distribution<unsigned> uidTree(0, featNum-1);
        std::uniform_real_distribution<float> urdTree(0, 1);

        for (int i = 0; i < nodeNum; i++){
            _fidxArr.ii[i] = i;
            // _fidxArr.ii[i] = uidTree(e);
            _fvalArr.vv[i] = urdTree(e);
        }
    }

    void initGroupChildren(int gDep, int gCol) {
        int gNumPre = gDep == 0 ? 0 : ((1 << (4*(gDep-1))) - 1) / (16 - 1);
        int iStart = gNumPre + gCol * gnodeNum;
        for (int i = iStart, iEnd = iStart + gnodeNum; i < iEnd; ++i) {
            _children[i - iStart] = i;
        }
    }

    inline float leafVal(int32_t offset) {
        // 7 = 1 + 2 + 4
        return _fvalArr.vv[offset + 7];
    }
    
    inline int32_t nextGroup(int32_t offset) {
        return _children[offset];
    }
    
    inline unsigned short predictGroup(float* smpValg, bool isLeaf = false) {
        nodeStat s;
        cmpFS(smpValg, s);
        return calcOffset(s, isLeaf);
    }

    float testTime_core(float* smpArr){
        // int Common::cycleNum = 1000000000;
        float volatile res = 0.f;
        nodeStat volatile s;
        // volatile unsigned short nodei = 0;
        FeatIdx fidxArr, a;
        FeatVal volatile smpValArr;
        // _mm512_prefetch_i32gather_ps(_fidxArr.i, smpArr, 4, _MM_HINT_T2);
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < Common::cycleNum; ++ i){
            // 1.test instruction parallel
            // __m512 volatile smpVal = _mm512_i32gather_ps(_fidxArr.i, smpArr, 4);
            // __m512 volatile smpVal1 = _mm512_i32gather_ps(_fidxArr.i, smpArr, 4);
            // __m512 volatile smpVal2 = _mm512_i32gather_ps(_fidxArr.i, smpArr, 4);

            // #pragma ivdep
            // #pragma vector always
            for(int j = 0; j < gnodeNum; ++ j) {
                smpValArr.vv[j] = smpArr[_fidxArr.ii[j]];
            }

            // 2.test other instructions
            // __m512 volatile smpVal = _mm512_load_ps(smpArr);
            // __m512 volatile smpVal = _mm512_permutexvar_ps(_fidxArr.i, smpValArr.v);
            
            // // 3.compare
            // s.m = _mm512_cmp_ps_mask(_fvalArr.v, smpVal, _CMP_LT_OS) << 1;
            // s.m = _mm512_cmp_ps_mask(_fvalArr.v, smpValArr.v, _CMP_LT_OS) << 1;
            
            // // 4.calculate 
            // volatile unsigned short _nodei = Common::lookup[s.mm[0]];
            // volatile unsigned short nodei = (((1 << _nodei) & s.mm[1]) != 0) + _nodei * 2;
        }
        auto end = std::chrono::system_clock::now();
        std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
        return res;
    }

private:



    inline void cmpFS(float* smpArr, nodeStat& s) {
        // FeatIdx a;
        SampleGroupT volatile smpVal = _mm512_i32gather_ps(_fidxArr.i, smpArr, 4);
        // SampleGroupT volatile smpVal1 = _mm512_i32gather_ps(a.i, smpArr, 4);
        
        s.m = _mm512_cmp_ps_mask(_fvalArr.v, smpVal, _CMP_LT_OS) << 1;
        // auto volatile _s = _mm512_cmp_ps_mask(_fvalArr.v, smpVal, _CMP_LT_OS) << 1;
    }

    inline unsigned short calcOffset(nodeStat& s, bool isLeaf = false) {
        unsigned short _offset = Common::lookup[s.mm[0]];
        if (isLeaf) return _offset;
        return (((1 << _offset) & s.mm[1]) != 0) + _offset * 2;
    }

    // 
    FeatIdx _fidxArr;
    FeatVal _fvalArr;
    // store offset in tree
    std::vector<int32_t> _children; 
};


class RegTree {
public:
    RegTree() : _depthN(8) {
        
        // group depth = 4
        _depthG = _depthN / 4;
        int gNum = ((1 << (4*_depthG)) - 1) / (16 - 1);
        _groups.resize(gNum);
        
        for (int i = 0; i < _depthG; ++i) {
            int gNumPre = i == 0 ? 0 : ((2 << (4*(i-1))) - 1) / (16 - 1);
            for (int j = 0; j < (1 << (4*i)); ++ j) {
                int gIdx = gNumPre + j;
                _groups[gIdx].initGroupData(smpLen, 16);
                if (i != _depthG-1) {
                    _groups[gIdx].initGroupChildren(i, j);
                }
            }
        }
    }

    float predictTree(float* smpArr) {
        int32_t idx = 0;
        for(int i = 0; i < _depthG - 1; ++ i) {
            int offset = _groups[idx].predictGroup(smpArr);
            idx = _groups[idx].nextGroup(offset);
        }
        int offset = _groups[idx].predictGroup(smpArr, true);
        return _groups[idx].leafVal(offset);
    }

    float testTime(float* smpArr){
        // float volatile res = 0.f;
        // auto start = std::chrono::system_clock::now();
        // for (int i = 0; i < Common::cycleNum; ++ i){
        //     res = _groups[0].predictGroup(smpArr);
        // }
        // auto end = std::chrono::system_clock::now();
        // std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
        
        // return res;
        return _groups[0].testTime_core(smpArr);
    }

    void loadModel() {}

  const std::vector<NodeGroup>& GetGroups() const { return _groups; }

private:
    std::vector<NodeGroup> _groups;
    int _depthN;
    int _depthG;
};


class GBTreeModel {
public:
    // void loadModel(Json const& in) = 0;
    GBTreeModel() : _treeNum(10) {
        for (int i = 0; i < _treeNum; ++ i) {
            _trees.push_back(std::unique_ptr<RegTree>(new RegTree()));
        }
    };
    
    float predictGBT(float* smpArr) {
        float res = 0;
        for(int i = 0; i < _treeNum; ++ i) {
            res += _trees[i]->predictTree(smpArr);
        }
        return res;
    }

    float testTime(float* smpArr) {
        // float res = 0.f;
        // auto start = std::chrono::system_clock::now();
        // for (int i = 0; i < Common::cycleNum; ++ i){
        //     res = _trees[0]->predictTree(smpArr);
        // }
        // auto end = std::chrono::system_clock::now();
        // std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
        // return res;
        return _trees[0]->testTime(smpArr);
    }

private:
    // GBTreeModelParam param;
    std::vector<std::unique_ptr<RegTree>> _trees;
    int _treeNum;

};
