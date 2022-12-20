class Optimizer:
    """LinearScheduler class with linear warmup and decay.
    """

    def __init__(self, optimizer, max_lr=1e-4, t1=500, t2=5000):
        """Initialize class."""
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.t1 = t1
        self.t2 = t2
        self.num_step = 0
        self.lr = 0.0

    def step(self):
        self.num_step += 1
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        if self.num_step <= self.t1:
            self.lr = self.max_lr * self.num_step / self.t1
        else:
            self.lr = self.max_lr * max((1.0 - (self.num_step - self.t1) / (self.t2 - self.t1)), 0)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
