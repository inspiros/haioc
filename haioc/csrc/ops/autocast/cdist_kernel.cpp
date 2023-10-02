#include "../cdist.h"

#include <ATen/autocast_mode.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            at::Tensor cdist_autocast(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p) {
                c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
                return haioc::ops::cdist(
                        at::autocast::cached_cast(at::kFloat, x1),
                        at::autocast::cached_cast(at::kFloat, x2),
                        p)
                        .to(at::result_type(x1, x2));
            }
        }

        TORCH_LIBRARY_IMPL(haioc, Autocast, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::cdist"),
                    TORCH_FN(cdist_autocast));
        }
    }
}
