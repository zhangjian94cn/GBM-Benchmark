
#pragma once

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
#include <tbb/tbb.h>

#include "common.h"
#include "pred_transform.h"

static const int gnodeNum = sizeof(__m512) / sizeof(int);
static const int smpLen = 16;
static const int gInnerDep = 4;

#define prefetch(x) __builtin_prefetch(x)

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
    uint8_t mm[sizeof(__mmask16) / sizeof(uint8_t)];
};

// 
class NodeGroup {

public:
    NodeGroup() {
        _children.resize(gnodeNum, 0);
        for (int i = 0; i < gnodeNum; ++ i) {
            _fidxArr.ii[i] = 0;
            _fvalArr.vv[i] = 0.f;
        }
    }
    NodeGroup(FeatIdx& _fia, FeatVal& _fiv, std::vector<int32_t>& _children) : 
            _fidxArr(_fia), _fvalArr(_fiv), _children(_children) {}

    void setNodeData(int nIdx, float fVal, int fIdx) {
        _fidxArr.ii[nIdx] = fIdx;
        _fvalArr.vv[nIdx] = fVal;
    }

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

    void initGroupChildren(int gDep, int gCol, bool leaf = false) {
        // int gNumPre = gDep == 0 ? 0 : ((1 << (4*(gDep-1))) - 1) / (16 - 1);
        int gNumPre = ((1 << (4 * (gDep+1))) - 1) / (16 - 1);
        if (leaf) gNumPre = 0;
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

    inline unsigned short predictGroup(const float* smpValg) {
        nodeStat s;
        // cmpFS(smpValg, s);
        
        __m512 smpVal = _mm512_i32gather_ps(_fidxArr.i, smpValg, 4);
        s.m = _mm512_cmp_ps_mask(_fvalArr.v, smpVal, _CMP_LT_OS) << 1;
        uint8_t _offset = Common::lookup[s.mm[0]];
        return (((1 << _offset) & s.mm[1]) != 0) + _offset * 2;
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



    inline void cmpFS(const float* smpArr, nodeStat& s) {
        // FeatIdx a;
        // __m512 smpVal;
        __m512 smpVal = _mm512_i32gather_ps(_fidxArr.i, smpArr, 4);
        s.m = _mm512_cmp_ps_mask(_fvalArr.v, smpVal, _CMP_LT_OS) << 1;
        
        // #pragma ivdep
        // #pragma vector always
        // for (int i = 0; i < 16; ++ i) {
        //     s.mm[i] = _fvalArr.vv[i] < smpArr[_fidxArr.ii[i]];
        // }
        // s.m = s.m << 1;

        // s.m = _mm512_cmp_ps_mask(_fvalArr.v, smpVal.v, _CMP_LT_OS) << 1;

        // SampleGroupT volatile smpVal = _mm512_i32gather_ps(_fidxArr.i, smpArr, 4);
        // SampleGroupT volatile smpVal1 = _mm512_i32gather_ps(a.i, smpArr, 4);
        
        // auto volatile _s = _mm512_cmp_ps_mask(_fvalArr.v, smpVal, _CMP_LT_OS) << 1;
    }

    inline unsigned short calcOffset(nodeStat& s, bool isLeaf = false) {
        unsigned short _offset = Common::lookup[s.mm[0]];
        // if (isLeaf) return _offset;
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
    RegTree() : _depthN(2) {
        
        // group depth = 4
        _depthG = _depthN / 4;
        int gNum = ((1 << (4 * _depthG)) - 1) / (16 - 1);
        _groups.resize(gNum);
        
        for (int i = 0; i < _depthG; ++i) {
            // int gNumPre = i == 0 ? 0 : ((2 << (4*(i-1))) - 1) / (16 - 1);
            int gNumPre = i == 0 ? 0 : ((1 << (4 * i)) - 1) / (16 - 1);
            for (int j = 0; j < (1 << (4*i)); ++ j) {
                int gIdx = gNumPre + j;
                _groups[gIdx].initGroupData(smpLen, 16);
                if (i != _depthG-1) {
                    _groups[gIdx].initGroupChildren(i, j);
                }
            }
        }
    }

    RegTree(int depthN, const std::vector<float>& weight, const std::vector<int>& index) {
        _depthN = depthN;
        _depthG = _depthN / 4;
        int gNum = ((1 << (4 * _depthG)) - 1) / (16 - 1);
        _groups.resize(gNum);

        int nIdx = 0;
        for (int i = 0; i < _depthG; ++i) {
            // previous group number
            int gNumPre = i == 0 ? 0 : ((1 << (4 * i)) - 1) / (16 - 1);
            // current group number
            int gNumCur = 1 << (4 * i);
            // node offset of current group row
            int nNumPre = 15 * gNumPre;
            // traverse current group by row
            for (int r = 0; r < gInnerDep; ++ r) {
                for (int j = 0; j < gNumCur; ++ j) {
                    // current group index 
                    int gIdxCur = gNumPre + j;
                    // previous node in a group
                    int rPre = r == 0 ? 0 : (1 << r) - 1;
                    // jPre represents the start idx of node in this line of group
                    int jPre = rPre * gNumCur + j * (1 << r);
                    // traverse current group's row
                    for (int n = 0; n < (1 << r); ++ n) {
                        nIdx = nNumPre + jPre + n;
                        _groups[gIdxCur].setNodeData(rPre + n, weight[nIdx], index[nIdx]);
                    }
                }
            }
            
            // init group children
            for (int j = 0; j < gNumCur; ++ j) {
                int gIdxCur = gNumPre + j;
                if (i != _depthG - 1) {
                    _groups[gIdxCur].initGroupChildren(i, j);
                }
                else {
                    _groups[gIdxCur].initGroupChildren(i, j, true);
                }
                // _groups[gIdxCur].initGroupChildren(i, j);
            }
        }

        _leaf.resize(1 << _depthN);
        for (int n = 0; n < (1 << _depthN); ++ n) {
            _leaf[n] = (weight[nIdx + n + 1]);
        }

    }

    // float predictTree(float* smpArr) {
    //     int32_t idx = 0;
    //     for(int i = 0; i < _depthG - 1; ++ i) {
    //         int offset = _groups[idx].predictGroup(smpArr);
    //         idx = _groups[idx].nextGroup(offset);
    //     }
    //     int offset = _groups[idx].predictGroup(smpArr, true);
    //     float leafVal = _groups[idx].leafVal(offset);
    //     return leafVal;
    // }

    inline float predictTree(const float* smpArr) {
        int32_t idx = 0, offset = 0;
        // for(int i = 0; i < 1; ++ i) {
        // prefetch(&_groups[0]);
        for(int i = 0; i < _depthG; ++ i) {
            offset = _groups[idx].predictGroup(smpArr);
            idx    = _groups[idx].nextGroup(offset);
            // offset = _groups[0].predictGroup(smpArr);
            // idx    = _groups[0].nextGroup(offset);
        }

        // offset = _groups[idx].predictGroup(smpArr);
        // idx    = _groups[idx].nextGroup(offset);
        
        // int offset = _groups[idx].predictGroup(smpArr, true);
        // float leafVal = _groups[idx].leafVal(offset);
        
        // return 0.0f;
        return _leaf[idx];
    }

    float testTimePre(float* smpArr){
        return _groups[0].testTime_core(smpArr);
    }

    float testTime(float* smpArr){
        float volatile res = 0.f;
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < Common::cycleNum; ++ i){
            res = _groups[0].predictGroup(smpArr);
        }
        auto end = std::chrono::system_clock::now();
        std::cout << (end-start).count()/1000000.0 << "ms" << std::endl;
        
        return res;
    }

    void loadModel() {}

  const std::vector<NodeGroup>& GetGroups() const { return _groups; }

private:
    std::vector<NodeGroup> _groups;
    std::vector<float> _leaf;
    int _depthN;
    int _depthG;
};


class GBTreeModel {
public:
    GBTreeModel() : _treeNum(0) {};
    
    void setTreeDepth(int depth) {
        _depth = depth;
    }
    
    int getTreeNum() { 
        return _treeNum; 
    };

    void pushTree(const std::vector<float>& weight, const std::vector<int>& index) {
        ++ _treeNum;
        // _trees.push_back(new RegTree(_depth, weight, index));
        _trees.push_back(RegTree(_depth, weight, index));
    }

    float predictGBT(const float* smpArr) {
        // float res[1000] = {0, };
        // tbb::parallel_for(0, _treeNum, 1, [&](int i) {
        // for(int i = 0; i < _treeNum; ++ i) {
        //     res[i] = _trees[i]->predictTree(smpArr);
        // }
        // });

        float res = 0.f;
        for(int i = 0; i < _treeNum; ++ i) {
            res += _trees[i].predictTree(smpArr);
        }
        return sigmoid(res);
    }

    float testTimePre(float* smpArr); 
    float testTime(float* smpArr); 

    std::vector<RegTree> _trees;

private:
    // GBTreeModelParam param;
    // std::vector<RegTree> _trees;
    // std::vector<RegTree*> _trees;
    int _treeNum;
    int _depth;
};
