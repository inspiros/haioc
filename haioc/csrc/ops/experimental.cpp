#include <torch/types.h>

#include "cpu/signum.h"

namespace haioc {
    namespace ops {
        namespace experimental {
            void _test_signum(const at::Scalar &input,
                             c10::optional<at::ScalarType> dtype) {
                auto dtype_ = dtype.value_or(at::kFloat);
                AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16,
                                          dtype_, "_test_signum", [&] {
                    std::cout << utils::signum(input.to<scalar_t>()) << std::endl;
                });
            }

            TORCH_LIBRARY_FRAGMENT(haioc, m) {
                m.def("haioc::_test_signum(Scalar input,  ScalarType? dtype=None) -> ()", TORCH_FN(_test_signum));
            }
        }
    }
}
