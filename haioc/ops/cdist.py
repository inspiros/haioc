import torch
from torch import Tensor

from ..extension import _assert_has_ops

__all__ = [
    'cdist',
]


def cdist(x1: Tensor,
          x2: Tensor,
          p: float = 2) -> Tensor:
    r"""Compute pairwise p-norm distance between :attr:`x1` and :attr:`x2`.

    Arguments:
        x1 (tensor): input tensor of shape :math:`B \times P \times M`.
        x2 (tensor): input tensor of shape :math:`B \times R \times M`.
        p (float): p value for the p-norm distance to calculate between each vector pair.
    """
    _assert_has_ops()
    return torch.ops.haioc.cdist(x1, x2, p)
