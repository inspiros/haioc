import math
import torch
from torch import Tensor
from tqdm import trange

import haioc
from test_utils import TimeMeter


def signum(x: float) -> float:
    return float(0 < x) - float(x < 0)


@torch.jit.script
def py_cdist(x1: Tensor, x2: Tensor, p: float = 2, backward: bool = False) -> Tensor:
    unbatched = x1.ndim == 2
    if unbatched:
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
    output = torch.empty(x1.size(0), x1.size(1), x2.size(1), dtype=x1.dtype, device=x1.device)
    r_p = 1 / p
    if backward:
        r_p -= 1

    n_kernels = output.numel()
    for index in range(n_kernels):
        j = index % x2.size(1)
        i = (index // x2.size(1)) % x1.size(1)
        b = index // (x2.size(1) * x1.size(1))

        val = 0.0
        for k in range(x1.size(2)):
            val += math.pow(math.fabs(x1[b, i, k].item() - x2[b, j, k].item()), p)
        output[b, i, j] = math.pow(val, r_p)

    return output if not unbatched else output.squeeze_(0)


@torch.jit.script
def py_cdist_backward(x1: Tensor, x2: Tensor, p: float = 2) -> Tensor:
    unbatched = x1.ndim == 2
    if unbatched:
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
    output = py_cdist(x1, x2, p, backward=True)
    grad_x1 = torch.zeros_like(x1)

    if p != 0:
        n_kernels = x1.size(0) * x1.size(1)
        for index in range(n_kernels):
            i = index % x1.size(1)
            b = index // x1.size(1)

            for j in range(x2.size(1)):
                for k in range(x1.size(2)):
                    val = x1[b, i, k].item() - x2[b, j, k].item()
                    grad_x1[b][i][k] += math.pow(math.fabs(val), p - 1) * output[b][i][j] * signum(val)

    if unbatched:
        grad_x1 = grad_x1.squeeze_(0)
    return grad_x1


def main(p=2, n_trials=1000, device='cuda'):
    x1 = torch.rand(50, 256, device=device, requires_grad=True)
    x2 = torch.rand(40, 256, device=device, requires_grad=True)
    zero_grad = lambda: torch.optim.Optimizer({x1, x2}, {}).zero_grad()

    torch_out = torch.cdist(x1, x2, p=p)
    torch_out.sum().backward()
    torch_grad_x1 = x1.grad.clone()
    torch_grad_x2 = x2.grad.clone()
    zero_grad()

    c_out = haioc.cdist(x1, x2, p=p)
    c_out.sum().backward()
    c_grad_x1 = x1.grad.clone()
    c_grad_x2 = x2.grad.clone()
    zero_grad()

    print('outputs match:', torch.allclose(torch_out, c_out))
    print('grads match:',
          torch.allclose(torch_grad_x1, c_grad_x1, rtol=1e-5, atol=1e-5) and
          torch.allclose(torch_grad_x2, c_grad_x2, rtol=1e-5, atol=1e-5))

    time_meter = TimeMeter()

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            torch.cdist(x1, x2, p=p)
        pbar.set_description(f'[torch] average_fps={time_meter.fps:.05f}')

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            haioc.cdist(x1, x2, p=p)
        pbar.set_description(f'[haioc] average_fps={time_meter.fps:.05f}')


if __name__ == '__main__':
    main()
