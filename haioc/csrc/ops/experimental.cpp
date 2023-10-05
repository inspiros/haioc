#include <ATen/Parallel.h>
#include <torch/types.h>

#include "cpu/atomic.h"
#include "cpu/signum.h"
#include "utils/dispatch.h"

namespace haioc {
    namespace ops {
        namespace experimental {
            void _test_signum(const at::Scalar &input,
                              c10::optional<at::ScalarType> dtype) {
                auto dtype_ = dtype.value_or(at::kFloat);
                AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                           dtype_, "_test_signum", ([&] {
                    std::cout << utils::signum(input.to<scalar_t>()) << std::endl;
                }));
            }

            namespace impl {
                template<bool use_parallel_for = true, typename scalar_t, typename index_t>
                static inline void _test_omp_impl(
                        index_t n_kernels,
                        const at::TensorAccessor<scalar_t, 3> input,
                        at::TensorAccessor<scalar_t, 3> output) {
                    if constexpr (use_parallel_for) {
                        at::parallel_for(0, n_kernels, 1, [&](int64_t begin, int64_t end) {
                            for (index_t index = begin; index < end; index++) {
                                index_t i = index % input.size(2);
                                index_t c = (index / input.size(2)) % input.size(1);
                                index_t b = index / (input.size(2) * input.size(1));

                                index_t j = i % output.size(2);
                                scalar_t val = input[b][c][i] * i;
                                cpuAtomicAdd(&output[b][c][j], val);
                            }
                        });
                    } else {
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
                        for (index_t index = 0; index < n_kernels; index++) {
                            index_t i = index % input.size(2);
                            index_t c = (index / input.size(2)) % input.size(1);
                            index_t b = index / (input.size(2) * input.size(1));

                            index_t j = i % output.size(2);
                            scalar_t val = input[b][c][i] * i;
                            cpuAtomicAdd(&output[b][c][j], val);
                        }
                    }
                }
            }

            at::Tensor _test_omp(bool use_parallel_for = true) {
                auto input = at::ones({1, 2, 2000});
                int64_t n = 16;
                auto output = at::zeros({input.size(0), input.size(1), n},
                                        input.options());

                int64_t n_kernels = input.numel();
                AT_DISPATCH_ALL_TYPES(input.scalar_type(), "_test_omp", ([&] {
                    HAIOC_DISPATCH_INDEX_TYPE_DEVICE(n_kernels, CPU, ([&] {
                        auto output_a = output.accessor<scalar_t, 3>();
                        HAIOC_DISPATCH_BOOL(use_parallel_for, ([&] {
                            impl::_test_omp_impl<use_parallel_for, scalar_t, index_t>(
                                    n_kernels,
                                    input.accessor<scalar_t, 3>(),
                                    output_a);
                        }));
                    }));
                }));
                return output;
            }

            TORCH_LIBRARY_FRAGMENT(haioc, m) {
                m.def("haioc::_test_signum(Scalar input,  ScalarType? dtype=None) -> ()", TORCH_FN(_test_signum));
                m.def("haioc::_test_omp(bool use_parallel_for=True) -> Tensor", TORCH_FN(_test_omp));
            }
        }
    }
}
