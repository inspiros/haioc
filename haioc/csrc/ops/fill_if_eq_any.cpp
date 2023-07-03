#include "fill_if_eq_any.h"

#include <torch/types.h>

namespace haioc {
    namespace ops {
        at::Tensor fill_if_eq_any(
                at::Tensor &input,
                const at::Tensor &other,
                const at::Scalar &fill_value,
                const bool inplace = false) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("haioc::fill_if_eq_any", "")
                    .typed<decltype(fill_if_eq_any)>();
            return op.call(
                    input,
                    other,
                    fill_value,
                    inplace);
        }

        namespace detail {
            at::Tensor _fill_if_eq_any_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    const at::Tensor &other) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("haioc::_fill_if_eq_any_backward", "")
                                .typed<decltype(_fill_if_eq_any_backward)>();
                return op.call(
                        grad_output,
                        input,
                        other);
            }
        }

        TORCH_LIBRARY_FRAGMENT(haioc, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "haioc::fill_if_eq_any(Tensor input, Tensor other, Scalar fill_value, bool inplace = False) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "haioc::_fill_if_eq_any_backward(Tensor grad_output, Tensor input, Tensor other) -> Tensor")
            );
        }
    } // namespace ops
} // namespace haioc
