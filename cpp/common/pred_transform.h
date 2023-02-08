#pragma once

#include <cmath>

inline float sigmoid(const float x) {
    return 1.f / (1.f + exp(-x));
}

// inline float sigmoid(float x) {
//     return 0.5 + 0.5 * tanh(x / 2);
// }

// inline float sigmoid(float x) {
//     return x / (1 + abs(x));
// }
