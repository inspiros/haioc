#include "cdist.h"

#include <torch/types.h>

namespace haioc {
    namespace ops {
        at::Tensor cdist(
                const at::Tensor &x1,
                const at::Tensor &x2,
                double p) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("haioc::cdist", "")
                    .typed<decltype(cdist)>();
            return op.call(x1, x2, p);
        }

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _cdist_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("haioc::_cdist_backward", "")
                                .typed<decltype(_cdist_backward)>();
                return op.call(grad, x1, x2, p);
            }
        }

        TORCH_LIBRARY_FRAGMENT(haioc, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "haioc::cdist(Tensor x1, Tensor x2, float p=2) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "haioc::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p) -> (Tensor grad_x1, Tensor grad_x2)")
            );
        }
    } // namespace ops
} // namespace haioc
