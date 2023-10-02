import math
import time

__all__ = ['TimeMeter']


class TimeMeter:
    def __init__(self):
        self.avg = self.count = 0
        self.start = self.end = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        elapsed_time = self.end - self.start
        self.avg = (self.avg * self.count + elapsed_time) / (self.count + 1)
        self.count += 1

    def reset(self):
        self.avg = self.count = 0
        self.start = self.end = None

    @property
    def fps(self):
        return 1. / self.avg if self.avg else math.nan
