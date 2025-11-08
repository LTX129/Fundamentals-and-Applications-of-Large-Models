import time, math, json, os
from collections import deque

class SmoothedValue:
    def __init__(self, window_size=100):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.total += value
        self.count += 1

    @property
    def median(self):
        d = sorted(self.deque)
        n = len(d)
        if n==0: return 0.0
        mid = n//2
        return (d[mid] if n%2==1 else 0.5*(d[mid-1]+d[mid]))

    @property
    def avg(self):
        return self.total / max(1,self.count)

class Checkpointer:
    def __init__(self, ckpt_dir="checkpoints"):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def save(self, state, name="last.pt"):
        path = os.path.join(self.ckpt_dir, name)
        import torch
        torch.save(state, path)
        return path

    def load(self, path):
        import torch
        return torch.load(path, map_location="cpu")
