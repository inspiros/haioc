#pragma once

#include "../utils/scalar_type_utils.h"

namespace utils {
    template<typename T>
    typename std::enable_if<c10::is_unsigned<T>::value, int>::type
    __forceinline constexpr signum(const T x) {
        return T(0) < x;
    }

    template<typename T>
    typename std::enable_if<c10::is_signed<T>::value, int>::type
    __forceinline constexpr signum(const T x) {
        return (T(0) < x) - (x < T(0));
    }
}
