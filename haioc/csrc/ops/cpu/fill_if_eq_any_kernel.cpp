#include <ATen/ATen.h>
#include <torch/library.h>

#include "cpu_helpers.h"
#include "../utils/dispatch.h"

namespace haioc {
    namespace ops {
        namespace {
            template<typename scalar_t, typename index_t>
            static void fill_if_eq_any_kernel_impl(
                    const at::TensorAccessor<scalar_t, 1> input,
                    const at::TensorAccessor<scalar_t, 1> other,
                    scalar_t fill_value,
                    at::TensorAccessor<scalar_t, 1> output) {
                CPU_1D_KERNEL_LOOP_T(i, input.size(0), index_t) {
                    for (index_t j = 0; j < other.size(0); j++) {
                        if (input[i] == other[j]) {
                            output[i] = fill_value;
                            continue;
                        }
                    }
                }
            }

            at::Tensor fill_if_eq_any_forward_kernel(
                    at::Tensor &input,
                    const at::Tensor &other,
                    const double fill_value,
                    const bool inplace) {
                at::CheckedFrom c = "any_eq_any_forward";
                auto args = {
                        at::TensorArg(input, "input", 1),
                        at::TensorArg(other, "other", 2)};
                at::checkAllSameType(c, args);

                const int64_t n_kernels = input.numel();
                at::Tensor output;
                if (!inplace)
                    output = input.clone();
                else
                    output = input;

                auto input_flatten = input.flatten();
                auto other_flatten = other.flatten();
                auto output_flatten = output.flatten();

                AT_DISPATCH_ALL_TYPES(
                        input.scalar_type(), "fill_if_eq_any_forward_cpu", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto output_accessor =
                                output_flatten.accessor<scalar_t, 1>();
                        fill_if_eq_any_kernel_impl<scalar_t, index_t>(
                                input_flatten.accessor<scalar_t, 1>(),
                                other_flatten.accessor<scalar_t, 1>(),
                                static_cast<scalar_t>(fill_value),
                                output_accessor);
                    }));
                }));
                return output;
            }

            at::Tensor fill_if_eq_any_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    const at::Tensor &other) {
                const int64_t n_kernels = grad_output.numel();
                at::Tensor grad_input = grad_output.clone();

                auto input_flatten = input.flatten();
                auto other_flatten = other.flatten();
                auto grad_input_flatten = grad_input.flatten();

                AT_DISPATCH_ALL_TYPES(
                        grad_output.scalar_type(), "fill_if_eq_any_backward_cpu", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto grad_input_accessor =
                                grad_input_flatten.accessor<scalar_t, 1>();
                        fill_if_eq_any_kernel_impl<scalar_t, index_t>(
                                input_flatten.accessor<scalar_t, 1>(),
                                other_flatten.accessor<scalar_t, 1>(),
                                static_cast<scalar_t>(0),
                                grad_input_accessor);
                    }));
                }));
                return grad_input;
            }
        }

        TORCH_LIBRARY_IMPL(haioc, CPU, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::fill_if_eq_any"),
                    TORCH_FN(fill_if_eq_any_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::_fill_if_eq_any_backward"),
                    TORCH_FN(fill_if_eq_any_backward_kernel));
        }
    }
}
