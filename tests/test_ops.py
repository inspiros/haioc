import torch

import haioc


def test_any_eq_any():
    print('[any_eq_any]')
    data = torch.randperm(70 * 10).sub_(70 * 10 // 2).view(70, 10).double()
    xs = torch.arange(0, 50).double()

    output = haioc.ops.any_eq_any(data, xs)
    print(output)


def test_fill_if_eq_any():
    print('[fill_if_eq_any]')
    data = torch.randperm(70 * 10).sub_(70 * 10 // 2).view(70, 10).double()
    data.requires_grad_(True)
    xs = torch.arange(0, 50).double()

    output = haioc.ops.fill_if_eq_any(data, -xs, 0)
    print(output)

    output.sum().backward()
    assert torch.equal(torch.stack(torch.where(output == 0)), torch.stack(torch.where(data.grad == 0)))

    # this will likely raise an exception, not sure why
    print('grad_correct:',
          torch.autograd.gradcheck(lambda inp: haioc.ops.fill_if_eq_any(inp, -xs, 0, False),
                                   inputs=(data,)))


def test_signum():
    torch.ops.haioc._test_signum(0.1, torch.bfloat16)
    torch.ops.haioc._test_signum(0.0, torch.bfloat16)
    torch.ops.haioc._test_signum(-0.0, torch.bfloat16)
    torch.ops.haioc._test_signum(-0.1, torch.bfloat16)


if __name__ == '__main__':
    # test_any_eq_any()
    # test_fill_if_eq_any()
    test_signum()
