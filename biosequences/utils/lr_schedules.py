'''Bla bla

'''
from torch.optim import lr_scheduler
from biosequences.utils._factory import _Factory

class ConstantLRAfterWarmup(lr_scheduler.LambdaLR):
    '''Bla bla

    '''
    def __init__(self, optimizer, n_warmup_steps):
        self.n_warmup_steps = n_warmup_steps

        super(ConstantLRAfterWarmup, self).__init__(
            optimizer=optimizer,
            lr_lambda=self.lr_lambda
        )

    def lr_lambda(self, step):
        if step < self.n_warmup_steps:
            return float(step) / float(max(1, self.n_warmup_steps))
        else:
            return 1.0

class LinearDecayLRAfterWarmup(lr_scheduler.LambdaLR):
    '''Bla bla

    '''
    def __init__(self, optimizer, n_warmup_steps, n_max_steps, f_lower_bound=0.0):
        self.n_warmup_steps = n_warmup_steps
        self.n_max_steps = n_max_steps
        self.f_lower_bound = f_lower_bound

        super(LinearDecayLRAfterWarmup, self).__init__(
            optimizer=optimizer,
            lr_lambda=self.lr_lambda
        )

    def lr_lambda(self, step):
        if step < self.n_warmup_steps:
            return float(step) / float(max(1, self.n_warmup_steps))
        else:
            fraction = float(self.n_max_steps - step) / float(max(1, self.n_max_steps - self.n_warmup_steps))
            fraction = min(fraction, self.f_lower_bound)
            return max(0.0, fraction)

class ConstantLRAfterWarmupBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self,
                 optimizer,
                 n_warmup_steps
                 ):
        self._instance = ConstantLRAfterWarmup(
            optimizer=optimizer,
            n_warmup_steps=n_warmup_steps
        )
        return self._instance

class LinearDecayLRAfterWarmupBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self,
                 optimizer,
                 n_warmup_steps,
                 n_max_steps,
                 f_lower_bound=0.0
                 ):
        self._instance = LinearDecayLRAfterWarmup(
            optimizer=optimizer,
            n_warmup_steps=n_warmup_steps,
            n_max_steps=n_max_steps,
            f_lower_bound=f_lower_bound
        )
        return self._instance

factory_lr_schedules = _Factory()
factory_lr_schedules.register_builder('constant lr after warmup', ConstantLRAfterWarmupBuilder())
factory_lr_schedules.register_builder('linear decay lr after warmup', LinearDecayLRAfterWarmupBuilder())