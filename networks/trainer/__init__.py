import functools

import torch

tduration_names = ["duration_train_data_fetch", "duration_for_back_ward", "duration_train_log"]
vduration_names = ["duration_vdata_fetch", "duration_vpredict", "duration_vlog"]
test_duration_names = ["duration_test_data_fetch", "duration_test_predict", "duration_test_log"]

def is_multiprocessing(n_workers):
    return n_workers >= 3

def _get_a_training_progress_bar(pbar_len):
    from tqdm import tqdm

    return tqdm(
        total=pbar_len, unit='批', desc=f'\r正在进行训练准备', ncols=100,
        bar_format='{desc}{n}/{total} | {elapsed}/{remaining} |{rate_fmt}{postfix}',
    )

def _before_training(trainer, *data_iters):
    if trainer.k == 1:
        print(f'\r本次训练位于设备{trainer.runtime_cfg["device"]}上')
        # 创建网络
        setattr(trainer, "module", trainer.net_builder.build(True))
        if not is_multiprocessing(trainer.n_workers):
            # 设置进度条
            pbar_len = trainer.n_epochs * functools.reduce(
                lambda x, y: len(x) + len(y), data_iters
            )
            setattr(trainer, "pbar", _get_a_training_progress_bar(pbar_len))
    else:
        if not hasattr(trainer, "module"):
            setattr(trainer, "module", trainer.net_builder.build(True))
        if not hasattr(trainer, "pbar"):
            setattr(trainer, "pbar", _get_a_training_progress_bar(0))

def _after_training(trainer, *args):
    if trainer.k == 1:
        if not is_multiprocessing(trainer.n_workers):
            trainer.module.deactivate()
            del trainer.pbar
        else:
            trainer.module.deactivate()
            result_conn = args[-1]
            result_conn.send(trainer.module)

def _prepare_train(fn):
    """
    Decorator for training preparation and cleanup.
    Allows user-defined operations before and after training.
    Usage:
        @prepare_train
        def train_fn(...): ...
    """

    @functools.wraps(fn)
    def wrapper(trainer, *args):
        # if trainer.k == 1:
        #     pbar_len = trainer.n_epochs * functools.reduce(
        #         lambda x, y: len(x) + len(y), args
        #     )
        #     # print(f'\r本次训练位于设备{trainer.runtime_cfg["device"]}上')
        #     # # 创建网络
        #     # setattr(trainer, "module", trainer.net_builder.build(True))
        #     # # 设置进度条
        #     # setattr(trainer, "pbar", get_a_training_progress_bar(pbar_len))
        #     _before_training(trainer, pbar_len)
        _before_training(trainer, *args)
        result = fn(trainer, *args)
        # if trainer.k == 1:
        #     # trainer.module.deactivate()
        #     # del trainer.pbar
        #     _after_training(trainer)
        _after_training(trainer, *args)
        return result

    return wrapper


def _prepare_valid(fn):
    """
    Decorator for training preparation and cleanup.
    Allows user-defined operations before and after training.
    Usage:
        @prepare_train
        def train_fn(...): ...
    """

    @functools.wraps(fn)
    @torch.no_grad()
    def wrapper(trainer, *args):
        trainer.module.eval()
        n_workers = trainer.n_workers
        if is_multiprocessing(n_workers):
            vlog_q, pbar_q, epoch = args[-3:]
            pbar_q.put(f"世代{epoch}验证开始")
            result = fn(trainer, *args)
            trainer.module.train()
            vlog_q.put(None)
            pbar_q.put(f"世代{epoch}验证完毕")
        else:
            epoch = args[-1]
            trainer.pbar.set_description(f"世代{epoch + 1}验证中")
            result = fn(trainer, *args)
            trainer.module.train()
            trainer.pbar.set_description(f"世代{epoch + 1}验证完毕")
        return result

    return wrapper

# from .trainer import Trainer
from .new_trainer import Trainer
