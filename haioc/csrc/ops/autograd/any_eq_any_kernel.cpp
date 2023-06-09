#include "../any_eq_any.h"

#include <torch/autograd.h>

namespace haioc {
    namespace ops {
        TORCH_LIBRARY_IMPL(haioc, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::any_eq_any"),
                    torch::autograd::autogradNotImplementedFallback());
        }
    }
}
