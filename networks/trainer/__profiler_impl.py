import time

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import record_function

from utils.accumulator import Accumulator
from utils.decorators import prepare
from utils.history import History


@prepare('train')
def profiling_impl(n_epochs, os, log_path, trainer, data_iter):
    # 提取训练器参数
    net = trainer.module
    criterion_a = trainer.criterion_a
    optimizer_s = net.optimizer_s
    scheduler_s = net.scheduler_s
    # 记录项设置
    loss_names = [f'train_{item}' for item in net.loss_names]
    criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
    lr_names = net.lr_names
    history = History(*(criteria_names + loss_names + lr_names))
    # 进行性能测试
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            # with_stack=True,
    ) as prof:
        for epoch in range(n_epochs):
            with record_function("epoch_switch_consume"):
                trainer.pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
                history.add(
                    lr_names, [
                        [param['lr'] for param in optimizer.param_groups]
                        for optimizer in optimizer_s
                    ]
                )
                # 记录批次训练损失总和，评价指标，样本数
                metric = Accumulator(len(loss_names + criteria_names) + 1)
            with record_function("epoch_total_consume"):
                for X, y in data_iter:
                    with torch.profiler.record_function("forward_and_backward"):
                        net.train()
                        pred, ls_es = net.forward_backward(X, y)
                    with torch.profiler.record_function("metric_calculation"):
                        with torch.no_grad():
                            num_examples = X.shape[0]
                            correct_s = []
                            for criterion in criterion_a:
                                correct = criterion(pred, y)
                                correct_s.append(correct)
                            metric.add(
                                *correct_s, *[ls * num_examples for ls in ls_es],
                                num_examples
                            )
                    # with torch.profiler.record_function("batch_switch_consume"):
                    trainer.pbar.update(1)
            with record_function("epoch_switch_consume"):
                for scheduler in scheduler_s:
                    scheduler.step()
                history.add(
                    criteria_names + loss_names,
                    [metric[i] / metric[-1] for i in range(len(metric) - 1)]
                )
        # prof.step()
    trainer.pbar.close()
    with open(os.path.join(
            log_path,
            f'{net.__class__.__name__.lower()}_profiling_{time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())}.txt'),
            mode='w') as file:
        table = prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=15)
        file.writelines(table)
        print(table)