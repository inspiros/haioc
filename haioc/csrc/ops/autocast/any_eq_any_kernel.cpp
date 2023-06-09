#include "../any_eq_any.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            at::Tensor any_eq_any_autocast(
                    const at::Tensor &input,
                    const at::Tensor &other) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return any_eq_any(
                        at::autocast::cached_cast(at::kFloat, input),
                        at::autocast::cached_cast(at::kFloat, other));
            }
        }

        TORCH_LIBRARY_IMPL(haioc, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::any_eq_any"),
                    TORCH_FN(any_eq_any_autocast));
        }
    }
}
