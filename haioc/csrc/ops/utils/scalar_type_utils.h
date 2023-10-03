#pragma once

#include <ATen/ScalarType.h>

namespace c10 {
    template<typename T>
    struct is_signed : std::is_signed<T> {
    };

    template<typename T>
    struct is_unsigned : std::is_unsigned<T> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt32>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt32>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt8>> : std::bool_constant<true> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QInt8>> : std::bool_constant<false> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt8>> : std::bool_constant<false> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt8>> : std::bool_constant<true> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt2x4>> : std::bool_constant<false> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt2x4>> : std::bool_constant<true> {
    };

    template<>
    struct is_signed<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt4x2>> : std::bool_constant<false> {
    };

    template<>
    struct is_unsigned<c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::QUInt4x2>> : std::bool_constant<true> {
    };
}
