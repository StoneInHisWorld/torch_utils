import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from queue import Queue, LifoQueue
from threading import Event
from typing import Iterable, List

import torch
from torch import nn
from tqdm import tqdm

from utils.accumulator import Accumulator
from utils.history import History
from multiprocessing import process


@torch.no_grad()
def train_log_impl(
        criterion_a,  # 计算相关
        history, lr_names, cri_los_names,  # 记录对象
        Q: Queue,  # 信号量相关
):
    metric = None
    while True:
        # 从队列中获取信号
        item = Q.get(True)
        # 如果收到了训练完成信号，则记录最后一个世代的数据后退出
        if type(item) == Event:
            # 记录最后一世代训练数据
            history.add(
                cri_los_names,
                [metric[i] / metric[-1] for i in range(len(metric) - 1)]
            )
            return
        # 若是获取到了学习率组，则认为完成了一个世代的迭代，刷新累加器并记录学习率
        elif type(item) == list:
            # 记录学习率
            history.add(lr_names, item)
            # 如果是首次记录，则创建累加器
            if metric is None:
                metric = Accumulator(len(cri_los_names) + 1)
            else:
                # 记录训练、数据
                history.add(
                    cri_los_names,
                    [metric[i] / metric[-1] for i in range(len(metric) - 1)]
                )
                metric.reset()
        # 若是获取到了网络参数、预测值、损失值、标签值
        # 则进行评价指标的计算以及损失值的累加
        elif type(item) == tuple:
            state_dict, pred, ls_es, y = item
            # 计算训练准确率数据
            correct_s = []
            for criterion in criterion_a:
                correct = criterion(pred, y)
                correct_s.append(correct)
            num_examples = y.shape[0]
            metric.add(
                *correct_s, *[ls * num_examples for ls in ls_es],
                num_examples
            )
        else:
            raise NotImplementedError(f'无法识别的信号{item}！')
        Q.task_done()


@torch.no_grad()
def train_and_valid_log_impl(
        net, valid_iter,  # 数据相关
        criterion_a,  # 计算相关
        history, lr_names, cri_los_names,  # 记录对象
        pbar,  # 进度条相关
        Q: Queue,  # 信号量相关
):
    """在多线程处理中，训练以及验证数据的记录实现。
    该方法会持续请求Q队列中的数据，直到获取到Event对象后退出，退出之前会进行最后一次历史记录。
    在记录时对History对象进行了原地更改，因此本方法只会返回None
    :param net: 网络结构
    :param valid_iter: 验证数据迭代器
    :param criterion_a: 准确率计算函数
    :param history: 历史记录
    :param lr_names: 学习率名称，用于记录每个世代的学习率
    :param cri_los_names: 评价指标和损失值名称，用于记录每个世代的评价指标和损失值组合
    :param pbar: 进度条
    :param Q: 数据交换队列，train_impl通过该队列传输需要记录的信息
    :return: history历史记录
    """
    # state_dict = None
    metric = Accumulator(len(cri_los_names) + 1)
    while True:
        # 从队列中获取信号
        item = Q.get(True)
        # 如果收到了训练完成信号，则记录最后一个世代的数据后退出
        if type(item) == Event:
            # # 进行验证
            # valid_log = net.test_(valid_iter, criterion_a, pbar)
            # # 记录最后一世代的训练和验证数据
            # history.add(
            #     cri_los_names + list(valid_log.keys()),
            #     [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
            #     list(valid_log.values())
            # )
            return
        # 若是获取到了学习率组，则认为完成了一个世代的迭代，刷新累加器并记录学习率
        elif type(item) == list:
            lr_group, state_dict = item
            # 记录学习率
            history.add(lr_names, lr_group)
            # 进行验证
            net.load_state_dict(state_dict)
            valid_log = net.test_(valid_iter, criterion_a, pbar)
            # 记录训练和验证数据
            history.add(
                cri_los_names + list(valid_log.keys()),
                [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
                list(valid_log.values())
            )
            metric.reset()
        # 若是获取到了网络参数、预测值、损失值、标签值
        # 则进行评价指标的计算以及损失值的累加
        elif type(item) == tuple:
            pred, ls_es, y = item
            # 计算训练准确率数据
            correct_s = []
            for criterion in criterion_a:
                correct = criterion(pred, y)
                correct_s.append(correct)
            num_examples = y.shape[0]
            metric.add(
                *correct_s, *[ls * num_examples for ls in ls_es],
                num_examples
            )
        else:
            raise NotImplementedError(f'无法识别的信号{item}！')
        del item
        Q.task_done()


def train_impl(
        net, data_iter,  # 数据和网络设置
        n_epochs, optimizer_s, lr_scheduler_s,  # 训练设置
        pbar,  # 进度条
        device,  # 数据设置
        Q: Queue, finish: Event  # 信号量
):
    non_blocking = net.device.type == 'cuda' and data_iter.pin_memory
    # 训练主循环
    for epoch in range(n_epochs):
        if finish.is_set():
            break
        pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
        # 世代主循环
        for X, y in data_iter:
            X, y = (
                X.to(device, non_blocking=non_blocking),
                y.to(device, non_blocking=non_blocking)
            )
            net.train()
            # 得到预测值以及损失组，即对应每个损失函数的损失值
            pred, ls_es = net.forward_backward(X, y)
            try:
                # 将网络参数、预测值、损失值、标签值作为信号放入队列中
                Q.put((pred, ls_es, y))
            except queue.Full:
                Q.join()
            # 完成了一批数据的计算，更新进度条
            pbar.update(1)
        # 队列中放入每个优化器的学习率参数组
        try:
            Q.put([
                [
                    [param['lr'] for param in optimizer.param_groups]
                    for optimizer in optimizer_s
                ],
                net.state_dict()
            ])
        except queue.Full:
            Q.join()
        # 更新学习率变化器组
        for scheduler in lr_scheduler_s:
            scheduler.step()
    finish.set()
    try:
        Q.put(finish)
    except queue.Full:
        Q.join()


class Trainer:

    def __init__(self,
                 module, optimizer_s: torch.optim.Optimizer or List[torch.optim.Optimizer],
                 lr_scheduler_s, acc_fn,
                 n_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
                 with_hook=False, hook_mute=True):
        self.module = module
        # 存储必要训练对象
        self.__optimizer_s = optimizer_s
        self.__lr_scheduler_s = lr_scheduler_s
        self.__criterion_a = acc_fn
        self.__n_epochs = n_epochs
        self.__ls_fn = ls_fn
        # 设置数据存放参数
        self.device = module.device
        # 处理hook机制
        self.with_hook = with_hook
        self.hook_mute = hook_mute
        pass

    def train_with_threads(self,
                           train_iter, valid_iter=None,
                           pbar=None, n_workers=2) -> History:
        """多线程神经网络训练函数。
        :param n_workers: 能够启用的最大处理机数
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        # 修复内存越占越多的问题
        Q = Queue(50)
        finish = Event()
        # 提取基本训练对象
        net = self.module
        n_epochs = self.__n_epochs
        criterion_a = self.__criterion_a if isinstance(self.__criterion_a, list) else [self.__criterion_a]
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
        # 学习率项
        lr_names = net.lr_names
        optimizer_s = self.__optimizer_s
        lr_scheduler_s = self.__lr_scheduler_s
        criterion_a = self.__criterion_a

        # warnings.warn("多线程训练会造成死锁，目前无法修复，将于将来版本后删除", DeprecationWarning)
        # 设置进度条
        if pbar is None:
            # 如果调用函数不提供进度条，则创建一个新的进度条
            if valid_iter is not None:
                pbar = tqdm(total=(len(train_iter) + len(valid_iter)) * n_epochs, unit='批', position=0,
                            desc=f'训练中……', mininterval=1)
            else:
                pbar = tqdm(total=(len(train_iter)) * n_epochs, unit='批', position=0,
                            desc=f'训练中……', mininterval=1)
        # 设置历史记录对象和累加对象
        history = History(*(criteria_names + loss_names + lr_names))
        # 使用线程池处理训练线程和记录线程
        executor = ThreadPoolExecutor(n_workers)
        train_future = executor.submit(
            train_impl,
            net, train_iter, n_epochs, optimizer_s, lr_scheduler_s,
            pbar, net.device,
            Q, finish
        )
        # 记录线程会因为是否指定验证迭代器而有所不同
        if valid_iter is not None:
            log_future = executor.submit(
                train_and_valid_log_impl,
                deepcopy(net), valid_iter,
                criterion_a,
                history, lr_names, criteria_names + loss_names,
                pbar,
                Q
            )
        else:
            log_future = executor.submit(
                train_log_impl,
                criterion_a,
                history, lr_names, criteria_names + loss_names,
                Q
            )
        # 实时监控各项任务的执行情况
        for future in as_completed([train_future, log_future]):
            try:
                future.result()
            except Exception as e:
                # 如果有任务抛出异常，则设置结束信号，关闭所有任务
                finish.set()
                executor.shutdown(wait=False)
                raise e
        pbar.close()
        return history

    def train_and_valid(self, train_iter, valid_iter, pbar=None) -> History:
        """
        神经网络训练函数。
        :param pbar: 提供外来进度条。此参数是为了适配k折训练
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        # 读取属性以便训练，并将优化器、规划器、评价计算器序列化
        net = self.module
        n_epochs = self.__n_epochs
        criterion_a = self.__criterion_a if isinstance(self.__criterion_a, list) else [self.__criterion_a]
        non_blocking = self.device.type == 'cuda' and train_iter.pin_memory
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
        # 学习率项
        lr_names = net.lr_names
        history = History(*(criteria_names + loss_names + lr_names))
        # 设置进度条
        if pbar is None:
            pbar = tqdm(total=(len(train_iter) + len(valid_iter)) * n_epochs, unit='批', position=0,
                        desc=f'训练中……', mininterval=1)
        for epoch in range(n_epochs):
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
            history.add(
                lr_names, [
                    [param['lr'] for param in optimizer.param_groups]
                    for optimizer in self.__optimizer_s
                ]
            )
            # 记录批次训练损失总和，评价指标，样本数
            metric = Accumulator(len(loss_names + criteria_names) + 1)
            # 训练主循环
            for X, y in train_iter:
                X, y = (
                    X.to(self.device, non_blocking=non_blocking),
                    y.to(self.device, non_blocking=non_blocking)
                )
                net.train()
                pred, ls_es = net.forward_backward(X, y)
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
                pbar.update(1)
            for scheduler in self.__lr_scheduler_s:
                scheduler.step()
            valid_log = net.test_(valid_iter, criterion_a, pbar)
            # 记录训练数据
            history.add(
                criteria_names + loss_names + list(valid_log.keys()),
                [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
                list(valid_log.values())
            )
        return history

    def train(self, train_iter) -> History:
        """神经网络训练函数。
        :return: 训练数据记录`History`对象
        """
        net = self.module
        n_epochs = self.__n_epochs
        criterion_a = self.__criterion_a if isinstance(self.__criterion_a, list) else [self.__criterion_a]
        non_blocking = self.device.type == 'cuda' and train_iter.pin_memory
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
        # 学习率项
        lr_names = net.lr_names
        history = History(*(criteria_names + loss_names + lr_names))
        with tqdm(total=len(train_iter) * n_epochs, unit='批', position=0,
                  desc=f'训练中……', mininterval=1) as pbar:
            for epoch in range(n_epochs):
                pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
                history.add(
                    lr_names, [
                        [param['lr'] for param in optimizer.param_groups]
                        for optimizer in self.__optimizer_s
                    ]
                )
                # 记录批次训练损失总和，评价指标，样本数
                metric = Accumulator(len(loss_names + criteria_names) + 1)
                # 训练主循环
                for X, y in train_iter:
                    X, y = (X.to(self.device, non_blocking=non_blocking),
                            y.to(self.device, non_blocking=non_blocking))
                    net.train()
                    pred, ls_es = net.forward_backward(X, y)
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
                    pbar.update(1)
                for scheduler in self.__lr_scheduler_s:
                    scheduler.step()
                # TODO:记录训练数据，可以细化到每个批次
                history.add(
                    criteria_names + loss_names,
                    [metric[i] / metric[-1] for i in range(len(metric) - 1)]
                )
            pbar.close()
        return history

    def train_with_k_fold(self, train_loaders_iter,
                          k: int = 10, n_workers=1) -> History:
        """
        使用k折验证法进行模型训练
        :param n_workers: 使用的处理机数量
        :param train_loaders_iter: 数据加载器供给，提供k折验证的每一次训练所需训练集加载器、验证集加载器
        :param k: 将数据拆分成k折，每一折轮流作验证集，余下k-1折作训练集
        :return: k折训练记录，包括每一折训练时的('train_l', 'train_acc', 'valid_l', 'valid_acc')
        """
        k_fold_history = None
        with tqdm(total=k * self.__n_epochs, position=0, leave=True, unit='批', mininterval=1) as pbar:
            for i, (train_iter, valid_iter) in enumerate(train_loaders_iter):
                pbar.set_description(f'\r训练折{pbar.n}……')
                # self.__data_iter = train_iter
                # 计算训练批次数
                pbar.total = k * self.__n_epochs * (len(train_iter) + len(valid_iter))
                if n_workers <= 1:
                    history = self.train_and_valid(train_iter, valid_iter, pbar)
                else:
                    history = self.train_with_threads(train_iter, valid_iter, pbar)
                if k_fold_history is None:
                    k_fold_history = history
                else:
                    k_fold_history += history
                pbar.set_description(f'\r折{i + 1}训练完毕')
        return k_fold_history

    def __deal_with_hook(self, net: Iterable[nn.Module]):
        self.__last_forward_output, self.__last_backward_data = {}, {}
        forward_handlers = []
        backward_handlers = []

        def __hook_forward_fn(module, input, output):
            if self.hook_mute:
                try:
                    self.__last_forward_output.pop(module)
                except Exception as _:
                    pass
            else:
                print(f'{module.__class__.__name__} FORWARD')
                try:
                    last_input, last_output = self.__last_forward_output.pop(module)
                except Exception as _:
                    pass
                else:
                    flag = True
                    for li, i in zip(last_input, input):
                        flag = torch.equal(li, i) and flag
                    print(f'input eq: {flag}')
                    flag = True
                    for lo, o in zip(last_output, output):
                        flag = torch.equal(lo, o) and flag
                    print(f'output eq: {flag}')
                    print('-' * 20)
            # 记录模块的梯度
            self.__last_forward_output[module] = input, output
            return output

        def __hook_backward_fn(module, grad_input, grad_output):
            if self.hook_mute:
                try:
                    last_input, last_output = self.__last_backward_data.pop(module)
                except Exception as _:
                    pass
                else:
                    for li, i in zip(last_input, grad_input):
                        if li is None or i is None:
                            print(f'{module.__class__.__name__} FORWARD None grad within {li} or {i}')
                    for lo, o in zip(last_output, grad_output):
                        if lo is None or o is None:
                            print(f'None grad within {lo} or {o}')
            else:
                print(f'{module.__class__.__name__} BACKWARD')
                try:
                    last_input, last_output = self.__last_backward_data.pop(module)
                except Exception as _:
                    pass
                else:
                    flag = True
                    for li, i in zip(last_input, grad_input):
                        if li is None or i is None:
                            print(f'{module.__class__.__name__} FORWARD None grad within {li} or {i}')
                        else:
                            flag = torch.equal(li, i) and flag
                            print(f'in_grad eq: {flag}')
                    flag = True
                    for lo, o in zip(last_output, grad_output):
                        if lo is None or o is None:
                            print(f'None grad within {lo} or {o}')
                        else:
                            flag = torch.equal(lo, o) and flag
                            print(f'out_grad eq: {flag}')
                    print('-' * 20)
            self.__last_backward_data[module] = grad_input, grad_output

        for m in net:
            forward_handlers.append(m.register_forward_hook(hook=__hook_forward_fn))
            backward_handlers.append(m.register_full_backward_hook(hook=__hook_backward_fn))
        return forward_handlers, backward_handlers

    def __enter__(self):
        if self.with_hook:
            self.__f_handles, self.__b_handles = self.__deal_with_hook(self.module)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_hook:
            for handle in self.__f_handles + self.__b_handles:
                handle.remove()
