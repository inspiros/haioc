#pragma once

#ifdef _WIN32
#if defined(haioc_EXPORTS)
#define HAIOC_API __declspec(dllexport)
#else
#define HAIOC_API __declspec(dllimport)
#endif
#else
#define HAIOC_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define VISION_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define VISION_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define VISION_INLINE_VARIABLE __attribute__((weak))
#endif
#endif