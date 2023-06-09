# from torch.types import Number
from typing import Union

import torch
from torch import Tensor

from ..extension import _assert_has_ops

__all__ = [
    'fill_if_eq_any',
]


def fill_if_eq_any(input: Tensor,
                   other: Tensor,
                   fill_value: Union[float, int],
                   inplace: bool = False) -> Tensor:
    r"""
    Fill elements of :attr:`input` which is equal to any element of :attr:`other`
    with `fill_value`.

    Arguments:
        input (tensor): 2D input tensor of shape (batch_size, in_features).
        other (tensor): 1D tensor to be compared.
        fill_value (number): value to be filled.
        inplace (bool): modify :attr:`input` or not. Defaults to ``False``.
    """
    _assert_has_ops()
    return torch.ops.haioc.fill_if_eq_any(input, other, fill_value, inplace)
