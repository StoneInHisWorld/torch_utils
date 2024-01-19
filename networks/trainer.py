from copy import deepcopy
from queue import Queue
from threading import Event

import torch
from torch import nn
from tqdm import tqdm

from utils.accumulator import Accumulator
from utils.history import History
from utils.thread import Thread

epoch_ending = 'end_of_an_epoch'


@torch.no_grad()
def train_log_impl(
        net, history, acc_fn,
        finish, Q
) -> History:
    metric = Accumulator(3)
    while not finish.is_set() or Q.unfinished_tasks > 0:
        item = Q.get()
        if item == epoch_ending:
            metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
            Q.task_done()
            continue
        ls, state_dict, X, y = item
        net.load_state_dict(state_dict)
        # 开始计算准确率数据
        correct = acc_fn(net(X), y)
        num_examples = X.shape[0]
        metric.add(ls * num_examples, correct, num_examples)
        # 记录训练数据
        history.add(
            ['train_l', 'train_acc'],
            [metric[0] / metric[2], metric[1] / metric[2]]
        )
        Q.task_done()
    return history


@torch.no_grad()
def __valid_impl(net, test_iter, acc_fn, ls_fn) -> tuple:
    net.eval()
    metric = Accumulator(3)
    for features, lbs in test_iter:
        preds = net(features)
        metric.add(acc_fn(preds, lbs), ls_fn(preds, lbs) * len(features), len(features))
    return metric[0] / metric[2], metric[1] / metric[2]


@torch.no_grad()
def train_and_valid_log_impl(
        net, valid_iter, acc_fn, ls_fn, history,
        finish, Q: Queue  # 信号量
) -> History:
    metric = Accumulator(3)
    while not finish.is_set() or Q.unfinished_tasks > 0:
        item = Q.get()
        if item == epoch_ending:
            metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
            Q.task_done()
            continue
        ls, state_dict, X, y = item
        net.load_state_dict(state_dict)
        # 开始计算验证数据
        valid_thread = Thread(net.__valid_impl, net, valid_iter, acc_fn, ls_fn)
        valid_thread.start()
        # 开始计算准确率数据
        correct = acc_fn(net(X), y)
        num_examples = X.shape[0]
        metric.add(ls * num_examples, correct, num_examples)
        # 记录训练、验证数据
        if valid_thread.is_alive():
            valid_thread.join()
        valid_acc, valid_l = valid_thread.get_result()
        history.add(
            ['train_l', 'train_acc', 'valid_l', 'valid_acc'],
            [metric[0] / metric[2], metric[1] / metric[2], valid_l, valid_acc]
        )
        Q.task_done()
    return history


def train_impl(
        net, data_iter, num_epochs, optimizer, ls_fn,
        Q: Queue, finish
):
    with tqdm(total=len(data_iter), unit='批', position=0,
              desc=f'训练中...', mininterval=1) as pbar:
        for epoch in range(num_epochs):
            pbar.reset(len(data_iter))
            pbar.set_description(f'世代{epoch + 1}/{num_epochs} 训练中...')
            # 训练主循环
            for X, y in data_iter:
                net.train()
                optimizer.zero_grad()
                ls = ls_fn(net(X), y)
                ls.backward()
                optimizer.step()
                Q.put((ls.item(), net.state_dict(), X, y))
                pbar.update(1)
            Q.put_nowait(epoch_ending)
        finish.set()
        pbar.set_description('正在进行收尾工作……')
        pbar.close()


class Trainer:

    def __init__(self,
                 module,
                 with_hook=False, hook_mute=True):
        self.module = module
        # 处理hook机制
        self.with_hook = with_hook
        self.hook_mute = hook_mute
        pass

    def train_and_valid(self,
                        data_iter, valid_iter, optimizer, acc_fn,
                        n_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
                        ) -> History:
        """
        神经网络训练函数。
        :param data_iter: 训练数据供给迭代器
        :param optimizer: 网络参数优化器
        :param n_epochs: 迭代世代
        :param ls_fn: 训练损失函数
        :param acc_fn: 准确率计算函数
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        net = self.module
        history = History('train_l', 'train_acc', 'valid_l', 'valid_acc')
        with tqdm(total=len(data_iter), unit='批', position=0,
                  desc=f'训练中...', mininterval=1) as pbar:
            for epoch in range(n_epochs):
                pbar.reset(len(data_iter))
                pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中...')
                metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
                # 训练主循环
                for X, y in data_iter:
                    net.train()
                    optimizer.zero_grad()
                    lo = ls_fn(net(X), y)
                    lo.backward()
                    optimizer.step()
                    with torch.no_grad():
                        correct = acc_fn(net(X), y)
                        num_examples = X.shape[0]
                        metric.add(lo.item() * num_examples, correct, num_examples)
                    pbar.update(1)
                # 记录训练数据
                pbar.set_description('验证中...')
                valid_acc, valid_l = net.test_(valid_iter, acc_fn, ls_fn)
                history.add(
                    ['train_l', 'train_acc', 'valid_l', 'valid_acc'],
                    [metric[0] / metric[2], metric[1] / metric[2], valid_l, valid_acc]
                )
            pbar.close()
        return history

    def train(self, data_iter, optimizer, acc_fn,
              n_epochs=10, ls_fn: nn.Module = nn.L1Loss()
              ) -> History:
        """
        神经网络训练函数。
        :param data_iter: 训练数据供给迭代器
        :param optimizer: 网络参数优化器
        :param n_epochs: 迭代世代
        :param ls_fn: 训练损失函数
        :param acc_fn: 准确率计算函数
        :return: 训练数据记录`History`对象
        """
        net = self.module
        history = History('train_l', 'train_acc')
        with tqdm(total=len(data_iter), unit='批', position=0,
                  desc=f'训练中...', mininterval=1) as pbar:
            for epoch in range(n_epochs):
                pbar.reset(len(data_iter))
                pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中...')
                metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
                # 训练主循环
                for X, y in data_iter:
                    net.train()
                    optimizer.zero_grad()
                    lo = ls_fn(net(X), y)
                    lo.backward()
                    optimizer.step()
                    with torch.no_grad():
                        correct = acc_fn(net(X), y)
                        num_examples = X.shape[0]
                        metric.add(lo.item() * num_examples, correct, num_examples)
                    pbar.update(1)
                # 记录训练数据
                history.add(
                    ['train_l', 'train_acc'],
                    [metric[0] / metric[2], metric[1] / metric[2]]
                )
            pbar.close()
        return history

    def train_with_k_fold(self, train_loaders_iter, optimizer, acc_fn,
                          n_epochs: int = 10, ls_fn: nn.Module = nn.L1Loss(),
                          k: int = 10, n_workers=1
                          ) -> History:
        """
        使用k折验证法进行模型训练
        :param n_workers:
        :param train_loaders_iter: 数据加载器供给，提供k折验证的每一次训练所需训练集加载器、验证集加载器
        :param optimizer: 优化器
        :param n_epochs: 迭代次数。数据集的总访问次数为k * num_epochs
        :param ls_fn: 损失函数
        :param k: 将数据拆分成k折，每一折轮流作验证集，余下k-1折作训练集
        :param acc_fn: 准确度函数
        :return: k折训练记录，包括每一折训练时的('train_l', 'train_acc', 'valid_l', 'valid_acc')
        """
        k_fold_history = History('train_l', 'train_acc', 'valid_l', 'valid_acc')
        with tqdm(range(k), position=0, leave=True, unit='fold') as pbar:
            for train_iter, valid_iter in train_loaders_iter:
                pbar.set_description(f'Training fold-{pbar.n}')
                if n_workers <= 1:
                    history = self.train_and_valid(train_iter, valid_iter, optimizer, acc_fn, n_epochs, ls_fn)
                else:
                    history = self.train_with_threads(train_iter, optimizer, acc_fn, n_epochs, ls_fn, valid_iter)
                k_fold_history += history
                pbar.update(1)
        return k_fold_history

    def train_with_threads(self,
                           data_iter, optimizer, acc_fn,
                           n_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
                           valid_iter=None) -> History:
        # TODO: 检查一下，为什么使用真实数量的数据集会引发死锁
        """
        神经网络训练函数。
        :param data_iter: 训练数据供给迭代器
        :param optimizer: 网络参数优化器
        :param n_epochs: 迭代世代
        :param ls_fn: 训练损失函数
        :param acc_fn: 准确率计算函数
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        Q = Queue()
        finish = Event()

        train_thread = Thread(
            train_impl,
            self.module, data_iter, n_epochs, optimizer, ls_fn,
            Q, finish
        )
        train_thread.start()
        if valid_iter is not None:
            history = History('train_l', 'train_acc', 'valid_l', 'valid_acc')
            log_thread = Thread(
                train_and_valid_log_impl,
                self.module, valid_iter, acc_fn, ls_fn, history,
                finish, Q
            )
        else:
            history = History('train_l', 'train_acc')
            log_thread = Thread(
                train_log_impl,
                self.module, history, acc_fn,
                finish, Q
            )
        log_thread.start()
        finish.wait()
        log_thread.join()
        return log_thread.get_result()

    def __deal_with_hook(self, net: nn.Module):
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_hook:
            for handle in self.__f_handles + self.__b_handles:
                handle.remove()
