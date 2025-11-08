import math
from torch.optim.lr_scheduler import _LRScheduler

class InverseSqrtWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps:int=4000, last_epoch:int=-1):
        self.warmup_steps = max(1, warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.warmup_steps ** 0.5) * min(step ** (-0.5), step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
