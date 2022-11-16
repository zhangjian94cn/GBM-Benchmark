#include <cmath>

inline float sigmoid(float x) {
    return 1.f / (1.f + exp(-x));
}

