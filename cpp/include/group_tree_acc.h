
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

// static const int gnodeNum = sizeof(__m512) / sizeof(int);
static const int smpLen = 16;
static const int gInnerDep = 4;

#define prefetch(x) __builtin_prefetch(x)

// class Base {
//     virtual ~Base() = 0;

// };

class Node {
public: 
    ~Node() = default;
    void load(float fval, uint8_t fidx)  {};

private:
    float   _fval;
    uint8_t _fidx;
};

// 
class Group {
public:
    Group() {
        _children.resize(16, 0);
        for (int i = 0; i < 16; ++ i) {
            _fidx.ii[i] = 0;
            _fval.vv[i] = 0.f;
        }
    }
    ~Group() {

    };
    /*!
    * \brief load the node group
    * \param fval feature value
    * \param fidx feature index
    */
    void load(std::vector<float> fval, std::vector<uint8_t> fidx) {
        for (int i = 0; i < fval.size(); ++ i) {
            _fval.vv[i] = fval[i];
            _fidx.ii[i] = fidx[i];
        }
    }
    // 
    void setNode(int nIdx, float fVal, int fIdx) {
        _fidx.ii[nIdx] = fIdx;
        _fval.vv[nIdx] = fVal;
    }
    /*!
    * \brief calculate the next node group offset
    * \param data test data
    */
    inline uint8_t next(const float* data) {
        nodeStat s;
        __m512 sval = _mm512_i32gather_ps(_fidx.i, data, 4);
        s.m = _mm512_cmp_ps_mask(_fval.v, sval, _CMP_LT_OS) << 1;
        uint8_t offset = Common::lookup[s.mm[0]];
        return _children[offset];
    }
    /*!
    * \brief calculate the next node group offset
    * \param data test data
    */
    inline uint8_t next(const register  __m512& data) {
        nodeStat s;
        __m512 sval = _mm512_permutexvar_ps(_fidx.i, data);
        s.m = _mm512_cmp_ps_mask(_fval.v, sval, _CMP_LT_OS) << 1;
        uint8_t offset = Common::lookup[s.mm[0]];
        return _children[offset];
    }

    void initChildren(int gDep, int gCol, bool leaf = false) {
        // int gNumPre = gDep == 0 ? 0 : ((1 << (4*(gDep-1))) - 1) / (16 - 1);
        int gNumPre = ((1 << (4 * (gDep+1))) - 1) / (16 - 1);
        if (leaf) gNumPre = 0;
        int iStart = gNumPre + gCol * gnodeNum;
        for (int i = iStart, iEnd = iStart + gnodeNum; i < iEnd; ++i) {
            _children[i - iStart] = i;
        }
    }


private:
    FeatIdxType _fidx;
    FeatValType _fval;
    std::vector<int8_t> _children; 
};

class Tree {
public:

    Tree(const std::vector<float>& weight, const std::vector<int>& index) {
        int _depthN = 8;
        int _depthG = _depthN / 4;
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
                        _groups[gIdxCur].setNode(rPre + n, weight[nIdx], index[nIdx]);
                    }
                }
            }
            
            // init group children
            for (int j = 0; j < gNumCur; ++ j) {
                int gIdxCur = gNumPre + j;
                if (i != _depthG - 1) {
                    _groups[gIdxCur].initChildren(i, j);
                }
                else {
                    _groups[gIdxCur].initChildren(i, j, true);
                }
            }
        }

        _leaf.resize(1 << _depthN);
        for (int n = 0; n < (1 << _depthN); ++ n) {
            _leaf[n] = (weight[nIdx + n + 1]);
        }

    }


    inline float predict(const float* s, const register __m512& r) {
        int8_t idx = 0;
        // idx = _groups[idx].next(s);
        idx = _groups[idx].next(r);
        idx = _groups[idx].next(s);
        return _leaf[idx];
    }

    void loadModel() {

    }

private:
    std::vector<Group> _groups;
    std::vector<float> _leaf;
    // int _depthG;
};


class TreeAgg {
public:

    TreeAgg( __m512i i):_i(i) {}

    void loadTree(
        // int tid, 
        const std::vector<float>& weight, 
        const std::vector<int>& index) {
        
        _treeAggs.push_back(Tree(weight, index));

    }

    inline void cache(const float* smp) {
        _r = _mm512_i32gather_ps(_i, smp, 4);
    }

    inline float predict(const float* smp, const register __m512& r) {
        float res = 0;
        const int size = _treeAggs.size();
        // cache(smp);
        // _r = _mm512_i32gather_ps(_i, smp, 4);
        for (int i = 0; i < size; ++ i) {
            res += _treeAggs[i].predict(smp, _r);
        }
        return res;
    };

    __m512i _i;

private:
    std::vector<Tree> _treeAggs;
    __m512  _r;
    // __m512i _i;
};


class GBTreeModel {
public:
    GBTreeModel() : _treeAggNum(0) {};
    
    void setTreeDepth(int depth) {
        _depth = depth;
    }
    
    int getTreeNum() { 
        return _treeAggNum; 
    };

    void pushTreeAgg(
        // __m512  r, 
        __m512i i, 
        const std::vector<std::vector<float>>& weight, 
        const std::vector<std::vector<int>>& index) {
        ++ _treeAggNum;
        // _treeAggs.push_back(new Tree(_depth, weight, index));
        TreeAgg _treeAgg = TreeAgg(i);
        for(int j = 0; j < 10; ++ j) {
            _treeAgg.loadTree(weight[j], index[j]);
        }
        _treeAggs.push_back(_treeAgg);
    }

    // float predictGBT(const float* smp) {
    //     float res = 0.f;
    //     for(int i = 0; i < _treeAggNum; ++ i) {
    //         res += _treeAggs[i].predict(smp);
    //     }
    //     return sigmoid(res);
    // }
    
    std::vector<TreeAgg> _treeAggs;

private:
    int _treeAggNum;
    int _depth;
};
