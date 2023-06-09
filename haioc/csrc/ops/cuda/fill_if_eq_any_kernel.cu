#include <ATen/ATen.h>
#include <torch/library.h>

#include "cuda_helpers.h"
#include "../utils/dispatch.h"

namespace haioc {
    namespace ops {
        namespace {
            inline unsigned int GET_THREADS() {
                return 1024;
            }

            template<typename scalar_t, typename index_t>
            static __global__ void fill_if_eq_any_kernel_impl(
                    const at::GenericPackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> input,
                    const at::GenericPackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits, index_t> other,
                    scalar_t fill_value,
                    at::GenericPackedTensorAccessor<scalar_t, 1, at::DefaultPtrTraits, index_t> output) {
                CUDA_1D_KERNEL_LOOP_T(i, input.size(0), index_t) {
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
                    const at::Scalar &fill_value,
                    const bool inplace) {
                at::CheckedFrom c = "any_eq_any_forward";
                auto args = {
                        at::TensorArg(input, "input", 1),
                        at::TensorArg(other, "other", 2)};
                at::checkAllSameType(c, args);
                at::checkAllSameGPU(c, args);

                at::cuda::CUDAGuard device_guard(input.get_device());
                const int64_t n_kernels = input.numel();
                at::Tensor output;
                if (!inplace)
                    output = input.clone();
                else
                    output = input;

                auto input_flatten = input.flatten();
                auto other_flatten = other.flatten();
                auto output_flatten = output.flatten();

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                AT_DISPATCH_ALL_TYPES(
                        input.scalar_type(), "fill_if_eq_any_forward_cuda", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto output_accessor =
                                output_flatten.generic_packed_accessor<scalar_t, 1, at::DefaultPtrTraits, index_t>();
                        fill_if_eq_any_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                input_flatten.generic_packed_accessor<scalar_t, 1, at::DefaultPtrTraits, index_t>(),
                                other_flatten.generic_packed_accessor<scalar_t, 1, at::RestrictPtrTraits, index_t>(),
                                fill_value.to<scalar_t>(),
                                output_accessor);
                    }));
                }));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return output;
            }

            at::Tensor fill_if_eq_any_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &input,
                    const at::Tensor &other) {
                at::cuda::CUDAGuard device_guard(grad_output.get_device());
                const int64_t n_kernels = grad_output.numel();
                at::Tensor grad_input = grad_output.clone();

                auto input_flatten = input.flatten();
                auto other_flatten = other.flatten();
                auto grad_input_flatten = grad_input.flatten();

                const unsigned int threads = GET_THREADS();
                const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                AT_DISPATCH_ALL_TYPES(
                        grad_output.scalar_type(), "fill_if_eq_any_backward_cuda", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                        auto grad_input_accessor =
                                grad_input_flatten.generic_packed_accessor<scalar_t, 1, at::DefaultPtrTraits, index_t>();
                        fill_if_eq_any_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                input_flatten.generic_packed_accessor<scalar_t, 1, at::DefaultPtrTraits, index_t>(),
                                other_flatten.generic_packed_accessor<scalar_t, 1, at::RestrictPtrTraits, index_t>(),
                                static_cast<scalar_t>(0),
                                grad_input_accessor);
                    }));
                }));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                return grad_input;
            }
        }

        TORCH_LIBRARY_IMPL(haioc, CUDA, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::fill_if_eq_any"),
                    TORCH_FN(fill_if_eq_any_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::_fill_if_eq_any_backward"),
                    TORCH_FN(fill_if_eq_any_backward_kernel));
        }
    }
}
