import os.path
import warnings
from copy import deepcopy
from threading import Event
from typing import Callable

import torch
import torch.nn as nn
from queue import Queue
from torch.nn import Module
from torch.utils import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_related.data_related import single_argmax_accuracy
from data_related.dataloader import LazyDataLoader
from utils.accumulator import Accumulator
from utils.func.torch_tools import init_wb
from utils.history import History
from utils.thread import Thread

epoch_ending = 'end_of_an_epoch'


class BasicNN(nn.Sequential):

    required_shape = (-1,)

    def __init__(self, *args: Module, **kwargs) -> None:
        """
        基本神经网络。提供神经网络的基本功能，包括权重初始化，训练以及测试。
        :param device: 网络所处设备
        :param init_meth: 网络初始化方法
        :param with_checkpoint: 是否使用检查点机制
        :param args: 需要添加的网络层
        """
        # 设置默认值
        init_meth = 'zero' if 'init_meth' not in kwargs.keys() else kwargs['init_meth']
        device = torch.device('cpu') if 'device' not in kwargs.keys() else kwargs['device']
        with_checkpoint = False if 'with_checkpoint' not in kwargs.keys() else kwargs['with_checkpoint']
        # 初始化各模块
        super().__init__(*args)
        self.apply(init_wb(init_meth))
        self.apply(lambda m: m.to(device))

        self.__device = device
        self.__last_backward_data = {}
        self.__last_forward_output = {}
        if with_checkpoint:
            warnings.warn('使用“检查点机制”虽然会减少前向传播的内存使用，但是会大大增加反向传播的计算量！')
        self.__checkpoint = with_checkpoint

    @torch.no_grad()
    def __train_log_impl(
            self, history, acc_fn,
            finish, Q
    ) -> History:
        metric = Accumulator(3)
        net = deepcopy(self)
        while not finish.is_set() or Q.unfinished_tasks > 0:
            item = Q.get()
            if item == epoch_ending:
                metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
                Q.task_done()
                # epoch += 1
                # i = 1
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

    @staticmethod
    @torch.no_grad()
    def __valid_impl(net, test_iter, acc_fn, ls_fn) -> tuple:
        net.eval()
        metric = Accumulator(3)
        for features, lbs in test_iter:
            preds = net(features)
            metric.add(acc_fn(preds, lbs), ls_fn(preds, lbs) * len(features), len(features))
        return metric[0] / metric[2], metric[1] / metric[2]

    @torch.no_grad()
    def __train_and_valid_log_impl(
            self, valid_iter, acc_fn, ls_fn, history,
            finish, Q: Queue  # 信号量
    ) -> History:
        metric = Accumulator(3)
        net = deepcopy(self)
        while not finish.is_set() or Q.unfinished_tasks > 0:
            item = Q.get()
            if item == epoch_ending:
                metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
                Q.task_done()
                continue
            ls, state_dict, X, y = item
            net.load_state_dict(state_dict)
            # 开始计算验证数据
            valid_thread = Thread(self.__valid_impl, net, valid_iter, acc_fn, ls_fn)
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
            # i += 1
            Q.task_done()
        return history

    def __train_impl(
            self, data_iter, num_epochs, optimizer, ls_fn,
            Q: Queue, finish
    ):
        with tqdm(total=len(data_iter), unit='批', position=0,
                  desc=f'训练中...', mininterval=1) as pbar:
            for epoch in range(num_epochs):
                pbar.reset(len(data_iter))
                pbar.set_description(f'世代{epoch + 1}/{num_epochs} 训练中...')
                # 训练主循环
                for X, y in data_iter:
                    self.train()
                    optimizer.zero_grad()
                    ls = ls_fn(self(X), y)
                    ls.backward()
                    optimizer.step()
                    Q.put_nowait((ls.item(), self.state_dict(), X, y))
                    pbar.update(1)
                Q.put_nowait(epoch_ending)
            finish.set()
            pbar.set_description('正在进行收尾工作……')
            pbar.close()

    def train_(self, data_iter, optimizer, num_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
               acc_fn=single_argmax_accuracy, valid_iter=None) -> History:
        """
        神经网络训练函数。
        :param data_iter: 训练数据供给迭代器
        :param optimizer: 网络参数优化器
        :param num_epochs: 迭代世代
        :param ls_fn: 训练损失函数
        :param acc_fn: 准确率计算函数
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        history = History('train_l', 'train_acc') if valid_iter is None else \
            History('train_l', 'train_acc', 'valid_l', 'valid_acc')
        with tqdm(total=len(data_iter), unit='批', position=0,
                  desc=f'训练中...', mininterval=1) as pbar:
            for epoch in range(num_epochs):
                pbar.reset(len(data_iter))
                pbar.set_description(f'世代{epoch + 1}/{num_epochs} 训练中...')
                metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
                # 训练主循环
                for X, y in data_iter:
                    self.train()
                    optimizer.zero_grad()
                    lo = ls_fn(self(X), y)
                    lo.backward()
                    optimizer.step()
                    with torch.no_grad():
                        correct = acc_fn(self(X), y)
                        num_examples = X.shape[0]
                        metric.add(lo.item() * num_examples, correct, num_examples)
                    pbar.update(1)
                # 记录训练数据
                if not valid_iter:
                    history.add(
                        ['train_l', 'train_acc'],
                        [metric[0] / metric[2], metric[1] / metric[2]]
                    )
                else:
                    pbar.set_description('验证中...')
                    valid_acc, valid_l = self.test_(valid_iter, acc_fn, ls_fn)
                    history.add(
                        ['train_l', 'train_acc', 'valid_l', 'valid_acc'],
                        [metric[0] / metric[2], metric[1] / metric[2], valid_l, valid_acc]
                    )
            pbar.close()
        return history

    hook_mute = False

    def train__(self, data_iter, optimizer, num_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
                acc_fn=single_argmax_accuracy, valid_iter=None) -> History:
        # TODO: 检查一下，为什么使用真实数量的数据集会引发死锁
        """
        神经网络训练函数。
        :param data_iter: 训练数据供给迭代器
        :param optimizer: 网络参数优化器
        :param num_epochs: 迭代世代
        :param ls_fn: 训练损失函数
        :param acc_fn: 准确率计算函数
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        Q = Queue()
        finish = Event()

        train_thread = Thread(
            self.__train_impl, data_iter, num_epochs, optimizer, ls_fn,
            Q, finish
        )
        train_thread.start()
        if valid_iter is not None:
            history = History('train_l', 'train_acc', 'valid_l', 'valid_acc')
            log_thread = Thread(
                self.__train_and_valid_log_impl,
                valid_iter, acc_fn, ls_fn, history,
                finish, Q
            )
        else:
            history = History('train_l', 'train_acc')
            log_thread = Thread(
                self.__train_log_impl,
                history, acc_fn,
                finish, Q
            )
        log_thread.start()
        finish.wait()
        log_thread.join()
        return log_thread.get_result()

    def hook_forward_fn(self, module, input, output):
        if not BasicNN.hook_mute:
            print(f'{module.__class__.__name__} FORWARD')
        try:
            last_input, last_output = self.__last_forward_output.pop(module)
        except Exception as _:
            pass
        else:
            flag = True
            for li, i in zip(last_input, input):
                flag = torch.equal(li, i) and flag
            if not BasicNN.hook_mute:
                print(f'input eq: {flag}')
            flag = True
            for lo, o in zip(last_output, output):
                flag = torch.equal(lo, o) and flag
            if not BasicNN.hook_mute:
                print(f'output eq: {flag}')
        self.__last_forward_output[module] = input, output
        if not BasicNN.hook_mute:
            print('-' * 20)

    def hook_backward_fn(self, module, grad_input, grad_output):
        if not BasicNN.hook_mute:
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
                    if not BasicNN.hook_mute:
                        print(f'in_grad eq: {flag}')
            flag = True
            for lo, o in zip(last_output, grad_output):
                if lo is None or o is None:
                    print(f'None grad within {lo} or {o}')
                else:
                    flag = torch.equal(lo, o) and flag
                    if not BasicNN.hook_mute:
                        print(f'out_grad eq: {flag}')
        self.__last_backward_data[module] = grad_input, grad_output
        if not BasicNN.hook_mute:
            print('-' * 20)

    def train_with_hook(self, data_iter, optimizer, num_epochs=10,
                        ls_fn: nn.Module = nn.L1Loss(),
                        acc_func=single_argmax_accuracy) -> History:
        """
        支持hook机制的训练过程方法。可以调用本方法对神经网络的前向反馈和后向反馈进行跟踪、监控。
        :param data_iter: 数据加载器
        :param optimizer: 网络优化器
        :param num_epochs: 迭代世代数
        :param ls_fn: 损失函数
        :param acc_func: 准确度计算函数
        :return: 训练过程数据记录表
        """
        for m in self:
            m.register_forward_hook(hook=BasicNN.hook_forward_fn)
            m.register_full_backward_hook(hook=BasicNN.hook_backward_fn)
        return self.train_(data_iter, optimizer, num_epochs, ls_fn, acc_func)

    def train_with_k_fold(self, train_loaders_iter, optimizer, num_epochs: int = 10,
                          ls_fn: nn.Module = nn.L1Loss(), k: int = 10,
                          acc_fn=single_argmax_accuracy) -> History:
        """
        使用k折验证法进行模型训练
        :param train_loaders_iter: 数据加载器供给，提供k折验证的每一次训练所需训练集加载器、验证集加载器
        :param optimizer: 优化器
        :param num_epochs: 迭代次数。数据集的总访问次数为k * num_epochs
        :param ls_fn: 损失函数
        :param k: 将数据拆分成k折，每一折轮流作验证集，余下k-1折作训练集
        :param acc_fn: 准确度函数
        :return: k折训练记录，包括每一折训练时的('train_l', 'train_acc', 'valid_l', 'valid_acc')
        """
        k_fold_history = History('train_l', 'train_acc', 'valid_l', 'valid_acc')
        with tqdm(range(k), position=0, leave=True, unit='fold') as pbar:
            for train_iter, valid_iter in train_loaders_iter:
                pbar.set_description(f'Training fold-{pbar.n}')
                history = self.train_(train_iter, optimizer, num_epochs, ls_fn, acc_fn, valid_iter=valid_iter)
                k_fold_history += history
                pbar.update(1)
        return k_fold_history

    @torch.no_grad()
    def test_(self, test_iter, acc_func=single_argmax_accuracy,
              loss: Callable = nn.L1Loss) -> [float, float]:
        """
        测试方法，取出迭代器中的下一batch数据，进行预测后计算准确度和损失
        :param test_iter: 测试数据迭代器
        :param acc_func: 计算准确度所使用的函数
        :param loss: 计算损失所使用的函数
        :return: 测试准确率，测试损失
        """
        self.eval()
        metric = Accumulator(3)
        for features, labels in test_iter:
            preds = self(features)
            metric.add(acc_func(preds, labels), loss(preds, labels) * len(features), len(features))
        return metric[0] / metric[2], metric[1] / metric[2]

    @torch.no_grad()
    def predict_(self, data_iter: DataLoader or LazyDataLoader,
                 unwrap_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None
                 ) -> torch.Tensor:
        self.eval()
        inputs, results, labels = [], [], []
        with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
            for fe_batch, lb_batch in data_iter:
                inputs.append(fe_batch)
                results.append(self(fe_batch))
                labels.append(lb_batch)
            data_iter.set_description('正对结果进行解包……')
        inputs = torch.cat(inputs, dim=0)
        results = torch.cat(results, dim=0)
        labels = torch.cat(labels, dim=0)
        if unwrap_fn is not None:
            results = unwrap_fn(inputs, results, labels)
        return results

    @property
    def device(self):
        return self.__device

    def load_state_dict_(self, where: str):
        assert os.path.exists(where), f'目录{where}无效！'
        assert where.endswith('.ptsd'), f'该文件{where}并非网络参数文件！'
        paras = torch.load(where)
        self.load_state_dict(paras)

    def __str__(self):
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device)

    def __call__(self, x):
        if self.__checkpoint:
            _check_first = False
            for m in self:
                can_check = _check_first and type(m) != nn.Dropout and type(m) != nn.BatchNorm2d
                x = checkpoint.checkpoint(m, x) if can_check else m(x)
                _check_first = True
            return x
        else:
            return super(BasicNN, self).__call__(x)
