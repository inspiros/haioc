#include "../fill_if_eq_any.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            class FillIfEqAnyFunction
                    : public torch::autograd::Function<FillIfEqAnyFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        torch::autograd::Variable &input,
                        const torch::autograd::Variable &other,
                        const double fill_value,
                        const bool inplace) {
                    at::AutoDispatchBelowADInplaceOrView g;
                    auto output = fill_if_eq_any(
                            input,
                            other,
                            fill_value,
                            inplace);

                    ctx->save_for_backward({input, other});

                    return {
                            output,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto input = saved[0];
                    auto other = saved[1];

                    auto grad_input = detail::_fill_if_eq_any_backward(
                            grad_output[0],
                            input,
                            other);

                    return {
                            grad_input,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };
        } // namespace

        at::Tensor fill_if_eq_any_autograd(
                at::Tensor &input,
                const at::Tensor &other,
                const double fill_value,
                const bool inplace) {
            return FillIfEqAnyFunction::apply(
                    input,
                    other,
                    fill_value,
                    inplace
            )[0];
        }

        TORCH_LIBRARY_IMPL(haioc, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::fill_if_eq_any"),
                    TORCH_FN(fill_if_eq_any_autograd));
        }
    }
}
