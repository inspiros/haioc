haioc ![GitHub](https://img.shields.io/github/license/inspiros/haioc) ![haioc](https://img.shields.io/badge/%C4%91mm-h%E1%BB%8Dc%20v%E1%BB%ABa%20th%C3%B4i-red)
======

This repo contains a small PyTorch's C++/Cuda extension of operations requested by [Hải](https://github.com/hocdot).
Current list of implemented operations:

| Name             | Differentiable |
|:-----------------|:--------------:|
| `any_eq_any`     |       ❌        |
| `fill_if_eq_any` |       ❌        |

## Installation

To install globally, clone this repo and run:

```
pip install .
```

Or build inplace binary with:

```
python setup.py build_ext --inplace
```

## Usage

### Example of `any_eq_any` and `fill_if_eq_any`:

```python
import torch

import haioc

data = torch.randperm(700 * 99).sub_(700 * 99 // 2).view(700, 99).int()
xs = torch.arange(0, 500).int()

delete_mask = haioc.any_eq_any(data, xs)
zero_mask = haioc.fill_if_eq_any(data, -xs, 0.)
```

Do check [`tests/test_haioc.py`](tests/test_haioc.py).

## License

The code is released under the MIT No Attribution license. See [`LICENSE.txt`](LICENSE.txt) for details.