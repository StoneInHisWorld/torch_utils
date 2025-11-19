import functools

from tqdm import tqdm


def prepare_train(fn):
    """
    Decorator for training preparation and cleanup.
    Allows user-defined operations before and after training.
    Usage:
        @prepare_train
        def train_fn(...): ...
    """
    def wrapper(trainer, *data_iter):
        setattr(trainer, "n_epochs", trainer.hps['epochs'])
        pbar_len = trainer.n_epochs * functools.reduce(lambda x, y: len(x) + len(y), data_iter)
        print(f'\r本次训练位于设备{trainer.runtime_cfg["device"]}上')
        # 创建网络
        setattr(trainer, "module", trainer.net_builder.build(True))
        # 设置进度条
        setattr(trainer, "pbar", tqdm(
            total=pbar_len, unit='批', desc=f'\r正在进行训练准备', ncols=100,
            bar_format='{desc}{n}/{total} | {elapsed}/{remaining} | {rate_fmt}{postfix}',
        ))
        result = fn(trainer, *data_iter)
        trainer.module.deactivate()
        del trainer.pbar, trainer.n_epochs, trainer.k
        return result

    return wrapper


# from .trainer import Trainer
from .new_trainer import Trainer

