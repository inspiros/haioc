import torch
from torch import Tensor

from ..extension import _assert_has_ops

__all__ = [
    'fill_if_eq_any',
]


def fill_if_eq_any(input: Tensor,
                   other: Tensor,
                   fill_value: float = 0.,
                   inplace: bool = False) -> Tensor:
    r"""

    Arguments:
        input (tensor): 2D input tensor of shape (batch_size, in_features).
        other (tensor): 1D tensor to be compared.
        fill_value (number): value to be filled. Defaults to 0.
        inplace (bool): modify ``input`` or not. Defaults to ``False``.
    """
    _assert_has_ops()
    return torch.ops.haioc.fill_if_eq_any(input, other, fill_value, inplace)
