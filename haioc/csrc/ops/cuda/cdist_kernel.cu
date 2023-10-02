#include <ATen/ATen.h>
#include <torch/library.h>

#include "cuda_helpers.h"
#include "signum.cuh"
#include "../utils/dispatch.h"

namespace haioc {
    namespace ops {
        namespace {
            constexpr unsigned int GET_THREADS() {
                return 1024;
            }

            template<bool renorm = true, typename scalar_t, typename index_t>
            static __global__ void cdist_kernel_impl(
                    index_t n_kernels,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                    scalar_t p,
                    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output) {
                scalar_t r_p = 1 / p;
                CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                    const index_t j = index % x2.size(1);
                    const index_t i = (index / x2.size(1)) % x1.size(1);
                    const index_t b = index / (x2.size(1) * x1.size(1));

                    scalar_t val = 0;
                    for (index_t k = 0; k < x1.size(2); k++) {
                        val += pow(abs(x1[b][i][k] - x2[b][j][k]), p);
                    }
                    if constexpr (renorm)
                        val = pow(val, r_p);
                    output[b][i][j] = val;
                }
            }

            template<bool negative = false, typename scalar_t, typename index_t>
            static __global__ void cdist_kernel_inf_impl(
                    int64_t n_kernels,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output) {
                CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                    index_t j = index % x2.size(1);
                    index_t i = (index / x2.size(1)) % x1.size(1);
                    index_t b = index / (x2.size(1) * x1.size(1));

                    scalar_t val = abs(x1[b][i][0] - x2[b][j][0]), tmp;
                    for (index_t k = 1; k < x1.size(2); k++) {
                        tmp = abs(x1[b][i][k] - x2[b][j][k]);
                        if constexpr (negative) {
                            if (tmp < val)
                                val = tmp;
                        } else {
                            if (tmp > val)
                                val = tmp;
                        }
                    }
                    output[b][i][j] = val;
                }
            }

            at::Tensor cdist_forward_kernel(
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p) {
                at::CheckedFrom c = "cdist_forward";
                auto args = {
                        at::TensorArg(x1, "x1", 1),
                        at::TensorArg(x2, "x2", 2)};
                at::checkAllSameGPU(c, args);
                at::checkAllSameType(c, args);

                bool unbatched = x1.ndimension() == 2;
                TORCH_CHECK(unbatched || x1.ndimension() == 3,
                            "x1 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || x2.ndimension() == 3,
                            "x2 must be 2-D (unbatched) or 3-D (batched) tensor.")
                TORCH_CHECK(unbatched || (x1.size(0) == x2.size(0)),
                            "batch_size of x1 and x2 do not match.")
                TORCH_CHECK((unbatched && x1.size(1) == x2.size(1)) ||
                            (!unbatched && x1.size(2) == x2.size(2)),
                            "feature dimension of x1 and x2 do not match.")

                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                if (unbatched) {
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels = batch_size * x1_c.size(1) * x2_c.size(1);
                auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)}, x1_c.options());

                if (p == 0) {
                    output.fill_(x1_c.size(2));
                } else {
                    const unsigned int threads = GET_THREADS();
                    const unsigned int blocks = GET_BLOCKS(threads, n_kernels);

                    AT_DISPATCH_FLOATING_TYPES(
                            x1_c.scalar_type(), "cdist_forward_cuda", ([&] {
                        HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                            auto output_accessor =
                                    output.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                            if (std::isinf(p)) {
                                HAIOC_DISPATCH_BOOL_NAME(negative, p < 0, ([&] {
                                    cdist_kernel_inf_impl<negative, scalar_t, index_t><<<blocks, threads>>>(
                                            n_kernels,
                                            x1_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                            x2_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                            output_accessor);
                                }));
                            } else {
                                cdist_kernel_impl<true, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                        x1_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        x2_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        static_cast<scalar_t>(p),
                                        output_accessor);
                            }
                        }));
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
                if (unbatched)
                    output.squeeze_(0);
                return output;
            }

            template<typename scalar_t, typename index_t>
            static __global__ void cdist_backward_x1_kernel_impl(
                    index_t n_kernels,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_output,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                    scalar_t p,
                    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_x1) {
                scalar_t r_p = 1 / p;
                CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                    const index_t i = index % x1.size(1);
                    const index_t b = index / x1.size(1);

                    scalar_t val;
                    for (index_t j = 0; j < x2.size(1); j++) {
                        for (index_t k = 0; k < x1.size(2); k++) {
                            val = x1[b][i][k] - x2[b][j][k];
                            grad_x1[b][i][k] += grad_output[b][i][j] *
                                                pow(abs(val), p - 1) / output[b][i][j] * utils::signum(val);
                        }
                    }
                }
            }

            template<typename scalar_t, typename index_t>
            static __global__ void cdist_backward_x2_kernel_impl(
                    index_t n_kernels,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_output,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> output,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x1,
                    const at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> x2,
                    scalar_t p,
                    at::GenericPackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits, index_t> grad_x2) {
                scalar_t r_p = 1 / p;
                CUDA_1D_KERNEL_LOOP_T(index, n_kernels, index_t) {
                    const index_t j = index % x2.size(1);
                    const index_t b = index / x2.size(1);

                    scalar_t val;
                    for (index_t i = 0; i < x1.size(1); i++) {
                        for (index_t k = 0; k < x1.size(2); k++) {
                            val = x2[b][j][k] - x1[b][i][k];
                            grad_x2[b][j][k] += grad_output[b][i][j] *
                                                pow(abs(val), p - 1) / output[b][i][j] * utils::signum(val);
                        }
                    }
                }
            }

            std::tuple<at::Tensor, at::Tensor> cdist_backward_kernel(
                    const at::Tensor &grad_output,
                    const at::Tensor &x1,
                    const at::Tensor &x2,
                    double p) {
                TORCH_WARN_ONCE("cdist_backward not working yet.")
                bool unbatched = x1.ndimension() == 2;

                auto grad_output_c = grad_output.contiguous();
                auto x1_c = x1.contiguous();
                auto x2_c = x2.contiguous();
                if (unbatched) {
                    grad_output_c = grad_output_c.unsqueeze(0);
                    x1_c = x1_c.unsqueeze(0);
                    x2_c = x2_c.unsqueeze(0);
                }

                int64_t batch_size = x1_c.size(0);
                int64_t n_kernels;
                auto grad_x1 = at::zeros_like(x1_c);
                auto grad_x2 = at::zeros_like(x2_c);

                if (p != 0) {
                    const unsigned int threads = GET_THREADS();
                    unsigned int blocks;

                    AT_DISPATCH_FLOATING_TYPES(
                            grad_output_c.scalar_type(), "cdist_backward_cuda", ([&] {
                        if (std::isinf(p)) {
                            n_kernels = grad_output_c.numel();
                            blocks = GET_BLOCKS(threads, n_kernels);
                            HAIOC_DISPATCH_INDEX_TYPE(std::max(n_kernels, batch_size *
                                                                          std::max(x1_c.size(1), x2_c.size(1))), ([&] {
                                HAIOC_DISPATCH_BOOL_NAME(negative, p < 0, ([&] {
                                    auto grad_x1_accessor =
                                            grad_x1.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                                    auto grad_x2_accessor =
                                            grad_x2.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                                    TORCH_CHECK_NOT_IMPLEMENTED(true, "")
//                                    cdist_backward_kernel_inf_impl<negative, scalar_t, index_t><<<blocks, threads>>>(
//                                            n_kernels,
//                                            grad_output_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
//                                            x1_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
//                                            x2_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
//                                            grad_x1_accessor,
//                                            grad_x2_accessor);
                                }));
                            }));
                        } else {
                            auto output = at::empty({batch_size, x1_c.size(1), x2_c.size(1)},
                                                    grad_output.options());
                            n_kernels = output.numel();
                            blocks = GET_BLOCKS(threads, n_kernels);
                            HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                                auto output_accessor =
                                        output.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                                cdist_kernel_impl<false, scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                        x1_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        x2_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        static_cast<scalar_t>(p),
                                        output_accessor);
                            }));

                            n_kernels = batch_size * x1.size(1);
                            blocks = GET_BLOCKS(threads, n_kernels);
                            HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                                auto grad_x1_accessor =
                                        grad_x1.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                                cdist_backward_x1_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                        grad_output_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        output.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        x1_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        x2_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        static_cast<scalar_t>(p),
                                        grad_x1_accessor);
                            }));

                            n_kernels = batch_size * x2.size(1);
                            blocks = GET_BLOCKS(threads, n_kernels);
                            HAIOC_DISPATCH_INDEX_TYPE(n_kernels, ([&] {
                                auto grad_x2_accessor =
                                        grad_x2.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>();
                                cdist_backward_x2_kernel_impl<scalar_t, index_t><<<blocks, threads>>>(
                                        n_kernels,
                                        grad_output_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        output.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        x1_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        x2_c.generic_packed_accessor<scalar_t, 3, at::RestrictPtrTraits, index_t>(),
                                        static_cast<scalar_t>(p),
                                        grad_x2_accessor);
                            }));
                        }
                    }));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
                if (unbatched) {
                    grad_x1.squeeze_(0);
                    grad_x2.squeeze_(0);
                }
                return std::make_tuple(grad_x1, grad_x2);
            }
        }

        TORCH_LIBRARY_IMPL(haioc, CUDA, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::cdist"),
                    TORCH_FN(cdist_forward_kernel));
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::_cdist_backward"),
                    TORCH_FN(cdist_backward_kernel));
        }
    }
}
