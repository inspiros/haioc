#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace haioc {
    namespace ops {
        HAIOC_API at::Tensor cdist(
                const at::Tensor &x1,
                const at::Tensor &x2,
                double p = 2);

        namespace detail {
            std::tuple<at::Tensor, at::Tensor> _cdist_backward(
                    const at::Tensor &grad,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p);
        }
    }
}
