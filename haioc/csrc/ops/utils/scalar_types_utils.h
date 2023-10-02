#pragma once

#include <ATen/ScalarType.h>

namespace std {
    // floating types
    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>> : bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>> : bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>> : bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>> : bool_constant<true> {
    };
}
