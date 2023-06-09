#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace haioc {
    namespace ops {
        HAIOC_API at::Tensor fill_if_eq_any(
                at::Tensor &input,
                const at::Tensor &other,
                const at::Scalar &fill_value,
                bool inplace);

        namespace detail {
            at::Tensor _fill_if_eq_any_backward(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    const at::Tensor &other);
        }
    }
}
