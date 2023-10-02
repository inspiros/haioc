#include "../cdist.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace haioc {
    namespace ops {
        namespace {
            class CdistFunction
                    : public torch::autograd::Function<CdistFunction> {
            public:
                static torch::autograd::Variable forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &x1,
                        const torch::autograd::Variable &x2,
                        double p) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    auto output = haioc::ops::cdist(x1, x2, p);

                    ctx->save_for_backward({x1, x2});
                    ctx->saved_data["p"] = p;

                    return output;
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto x1 = saved[0];
                    auto x2 = saved[1];
                    double p = ctx->saved_data["p"].toDouble();

                    auto grads = detail::_cdist_backward(
                            grad_output[0], x1, x2, p);
                    auto grad_x1 = std::get<0>(grads);
                    auto grad_x2 = std::get<1>(grads);

                    return {
                            grad_x1,
                            grad_x2,
                            torch::autograd::Variable(),
                    };
                }
            };
        } // namespace

        at::Tensor cdist_autograd(
                const at::Tensor &x1,
                const at::Tensor &x2,
                double p) {
            return CdistFunction::apply(x1, x2, p);
        }

        TORCH_LIBRARY_IMPL(haioc, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("haioc::cdist"),
                    TORCH_FN(cdist_autograd));
        }
    }
}
