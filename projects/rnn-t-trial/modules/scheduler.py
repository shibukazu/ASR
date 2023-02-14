import torch
from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):
    """TransformerLR class for adjustment of learning rate.

    The scheduling is based on the method proposed in 'Attention is All You Need'.
    """

    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1, verbose=False):
        """Initialize class."""
        self.warmup_steps = warmup_steps
        self.normalize = d_model**0.5
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return adjusted learning rate."""
        step = self.last_epoch + 1
        scale = self.normalize * min(step**-0.5, step * self.warmup_steps**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


class NoamScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
