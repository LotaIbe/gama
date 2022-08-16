import numpy as np
from river import stats


class StreamStats:
    def __init__(self, k):
        self.mean = stats.RollingMean(window_size=k)
        self.variance = stats.RollingVar(window_size=k)

    def update(self, element):
        self.variance = self.variance.update(element)
        self.mean = self.mean.update(element)

    def safe_div(self, a, b):
        return a / b if b else 0

    def validate_y(self, y):

        if not - 1000 <= self.safe_div((y - self.mean.get()), np.sqrt(self.variance.get())) <= 1000:
            # print(
            #     f"Bad value of target: {y} detected, replacing with {self.mean.get()} for drift detection algorithm")
            self.update(self.mean.get())
            return self.mean.get()
        else:
            self.update(y)
            return y
