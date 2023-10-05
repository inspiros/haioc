import torch
import haioc
from tqdm import trange

from test_utils import TimeMeter


def main(n_trials=3000):
    time_meter = TimeMeter()

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            torch.ops.haioc._test_omp(use_parallel_for=False)
        pbar.set_description(f'[omp parrallel for] average_fps={time_meter.fps:.05f}')

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            torch.ops.haioc._test_omp(use_parallel_for=True)
        pbar.set_description(f'[at::parallel_for] average_fps={time_meter.fps:.05f}')


if __name__ == '__main__':
    main()
