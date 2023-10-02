#pragma once

#include "../utils/scalar_types_utils.h"

namespace utils {
    template<typename T>
    typename std::enable_if<std::is_unsigned<T>::value, int>::type
    __forceinline __device__ constexpr signum(const T x) {
        return T(0) < x;
    }

    template<typename T>
    typename std::enable_if<std::is_signed<T>::value, int>::type
    __forceinline __device__ constexpr signum(const T x) {
        return (T(0) < x) - (x < T(0));
    }
}
