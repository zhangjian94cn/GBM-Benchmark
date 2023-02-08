
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

class Base {
    virtual ~Base() = 0;

};

class Node : Base{
public: 
    ~Node() = default;
    void load(float fval, uint8_t fidx)  {};

private:
    float   _fval;
    uint8_t _fidx;
};

// 
class Group : Base {
public:
    ~Group() = default;
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
    /*!
    * \brief calculate the next node group offset
    * \param data test data
    */
    inline uint8_t next(float* data) {
        nodeStat s;
        FeatValType sval = _mm512_i32gather_ps(_fidx.i, data, 4);
        s.m = _mm512_cmp_ps_mask(_fval.v, sval, _CMP_LT_OS) << 1;
        uint8_t offset = Common::lookup[s.mm[0]];
        return _children[offset]
    }
    /*!
    * \brief calculate the next node group offset
    * \param data test data
    */
    inline uint8_t next(FeatValType& data) {
        FeatValType sval = _mm512_permutexvar_ps(_fidx.i, data);
        s.m = _mm512_cmp_ps_mask(_fval.v, sval, _CMP_LT_OS) << 1;
        uint8_t offset = Common::lookup[s.mm[0]];
        return _children[offset];
    }

private:
    FeatIdxType _fidx;
    FeatValType _fval;
    std::vector<int8_t> _children; 
};

class Tree {
public:

    inline float predict(const float* s, const FeatValType& r) {
        const int32_t idx = 0;
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

    TreeAgg() {
        
    }

    void load() {
        
    }

    inline void cache(const float* smp) {
        _r = _mm512_i32gather_ps(_i, smp, 4);
    }

    inline float predict(const float* smp) {
        float res = 0;
        const int size = _treeAggs.size();
        for (int i = 0; i < size; ++ i) {
            res += _treeAggs[i].predict(smp, _r);
        }
        return res;
    };

private:
    std::vector<RegTree> _treeAggs;
    FeatValType _r;
    FeatIdxType _i;
}


class GBTreeModel {
public:
    GBTreeModel() : _treeAggNum(0) {};
    
    void setTreeDepth(int depth) {
        _depth = depth;
    }
    
    int getTreeNum() { 
        return _treeAggNum; 
    };

    void pushTreeAgg(const std::vector<float>& weight, const std::vector<int>& index) {
        ++ _treeAggNum;
        // _treeAggs.push_back(new Tree(_depth, weight, index));
        _treeAggs.push_back(TreeAgg(_depth, weight, index));
    }

    float predictGBT(const float* smp) {
        float res = 0.f;
        for(int i = 0; i < _treeAggNum; ++ i) {
            res += treeAggs[i].predict(smp);
        }
        return sigmoid(res);
    }
    
    std::vector<TreeAgg> _treeAggs;

private:
    int _treeAggNum;
    int _depth;
};
