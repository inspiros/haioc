import math
import torch
from torch import Tensor
from tqdm import trange

import haioc
from test_utils import TimeMeter


def py_cdist(x1: Tensor, x2: Tensor, p: float = 2) -> Tensor:
    unbatched = x1.ndim == 2
    if unbatched:
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
    output = torch.empty(x1.size(0), x1.size(1), x2.size(1), dtype=x1.dtype, device=x1.device)

    n_kernels = output.numel()
    for index in range(n_kernels):
        j = index % x2.size(1)
        i = (index // x2.size(1)) % x1.size(1)
        b = index // (x2.size(1) * x1.size(1))

        val = 0
        for k in range(x1.size(2)):
            val += math.pow(math.fabs(x1[b, i, k].item() - x2[b, j, k].item()), p)
        output[b, i, j] = math.pow(val, 1 / p)

    return output if not unbatched else output.squeeze_(0)


def main(p=2, n_trials=1000, device='cuda'):
    x1 = torch.rand(50, 128, device=device, requires_grad=True)
    x2 = torch.rand(40, 128, device=device, requires_grad=True)
    zero_grad = lambda: torch.optim.Optimizer({x1, x2}, {}).zero_grad()

    torch_out = torch.cdist(x1, x2, p=p)
    # torch_out.sum().backward()
    # torch_grad_x1 = x1.grad.clone()
    # torch_grad_x2 = x2.grad.clone()
    zero_grad()

    c_out = haioc.cdist(x1, x2, p=p)
    # c_out.sum().backward()
    # c_grad_x1 = x1.grad.clone()
    # c_grad_x2 = x2.grad.clone()
    zero_grad()

    print('outputs match:', torch.allclose(torch_out, c_out))
    # print(torch_grad_x1)
    # print(c_grad_x1)

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
