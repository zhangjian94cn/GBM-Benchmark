#pragma once
#include <cstdint>
#include <immintrin.h>

#if defined(__INTEL_COMPILER)
    #define PRAGMA_IVDEP            _Pragma("ivdep")
    #define PRAGMA_NOVECTOR         _Pragma("novector")
    #define PRAGMA_VECTOR_ALIGNED   _Pragma("vector aligned")
    #define PRAGMA_VECTOR_UNALIGNED _Pragma("vector unaligned")
    #define PRAGMA_VECTOR_ALWAYS    _Pragma("vector always")
    #define PRAGMA_ICC_TO_STR(ARGS) _Pragma(#ARGS)
    #define PRAGMA_ICC_OMP(ARGS)    PRAGMA_ICC_TO_STR(omp ARGS)
    #define PRAGMA_ICC_NO16(ARGS)   PRAGMA_ICC_TO_STR(ARGS)
    #define DAAL_TYPENAME           typename
#elif defined(__GNUC__)
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_ICC_TO_STR(ARGS)
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
    #define DAAL_TYPENAME typename
#elif defined(_MSC_VER)
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_ICC_TO_STR(ARGS)
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
    #define DAAL_TYPENAME
#else
    #define PRAGMA_IVDEP
    #define PRAGMA_NOVECTOR
    #define PRAGMA_VECTOR_ALIGNED
    #define PRAGMA_VECTOR_UNALIGNED
    #define PRAGMA_VECTOR_ALWAYS
    #define PRAGMA_ICC_OMP(ARGS)
    #define PRAGMA_ICC_NO16(ARGS)
    #define DAAL_TYPENAME typename
#endif

static const int gnodeNum = sizeof(__m512) / sizeof(int);

struct Common {
    static const uint8_t lookup[256];
    static const int cycleNum;
};

// union FeatIdx {
//     __m512i i;
//     int32_t ii[gnodeNum];
// };

// union FeatVal {
//     __m512  v;
//     float vv[gnodeNum];
// };

// union nodeStat {
//     __mmask16  m;
//     uint8_t mm[sizeof(__mmask16) / sizeof(uint8_t)];
// };

union FeatIdxType {
    __m512i i;
    int32_t ii[gnodeNum];
};

union FeatValType {
    __m512  v;
    float vv[gnodeNum];
};

union nodeStatType {
    __mmask16  m;
    uint8_t mm[sizeof(__mmask16) / sizeof(uint8_t)];
};


