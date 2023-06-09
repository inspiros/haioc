#include "../fill_if_eq_any.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            at::Tensor fill_if_eq_any_autocast(
                    at::Tensor &input,
                    const at::Tensor &other,
                    const double fill_value,
                    const bool inplace) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return fill_if_eq_any(
                        at::autocast::cached_cast(at::kFloat, input),
                        at::autocast::cached_cast(at::kFloat, other),
                        fill_value,
                        inplace)
                        .to(input.scalar_type());
            }
        }

        TORCH_LIBRARY_IMPL(haioc, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::fill_if_eq_any"),
                    TORCH_FN(fill_if_eq_any_autocast));
        }
    }
}
