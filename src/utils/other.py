import pandas as pd
from collections import defaultdict
import time
import numpy as np


def batch_list(arr, num_batches):
    out = []
    b = []
    for i in arr:
        if len(b) == num_batches:
            out.append(b)
            b = []
        b.append(i)
    if len(b) > 0: out.append(b)
    return out


class Timer:
    def __init__(self):
      self.D = defaultdict(lambda: [])
    def time(self, name):
      self.name = name
      return self
    def __enter__(self):
      self.t0 = time.time()
    def __exit__(self, exc_type, exc_val, exc_tb):
      self.D[self.name].append(time.time() - self.t0)
    def get_means_df(self):
      d = {f'{k} ({len(v)})':[np.sum(v)] for k,v in self.D.items()}
    #   d.update({f'{k} (total)':[np.sum(v)] for k,v in self.D.items()})
      return pd.DataFrame(d)