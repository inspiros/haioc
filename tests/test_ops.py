import torch
import haioc


def main():
    data = torch.randperm(700 * 99).sub_(700 * 99 // 2).view(700, 99).int()
    xs = torch.arange(0, 500).int()

    output = haioc.ops.any_eq_any(data, xs)
    print(output)


if __name__ == '__main__':
    main()
