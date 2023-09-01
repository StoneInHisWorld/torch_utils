import warnings
from typing import Callable, Any, Iterable

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils import checkpoint
from tqdm import tqdm

from utils.data_related import single_argmax_accuracy
from utils.history import History
from utils.tools import init_wb
from utils.accumulator import Accumulator


class BasicNN(nn.Sequential):
    required_shape = (-1,)

    # def __init__(self, device, *args: Module) -> None:
    def __init__(self, device: torch.device = torch.device('cpu'), init_meth: str = 'xavier',
                 with_checkpoint: bool = False, *args: Module) -> None:
        """
        基本神经网络。提供神经网络的基本功能，包括权重初始化，训练以及测试。
        :param device: 网络所处设备
        :param args: 需要添加的网络层
        :param device: 本网络所处设备
        :param init_meth: 网络初始化方法
        :param with_checkpoint: 是否使用检查点机制
        :param args: 输入网络的模型
        """
        super().__init__(*args)
        self.apply(init_wb(init_meth))
        # self.__init_wb(init_meth)
        self.apply(lambda m: m.to(device))

        self.__device = device
        self.__last_backward_data = {}
        self.__last_forward_output = {}
        if with_checkpoint:
            warnings.warn('使用“检查点机制”虽然会减少前向传播的内存使用，但是会大大增加训练计算量！')
        self.__checkpoint = with_checkpoint

    def __str__(self):
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device)

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
                    with torch.enable_grad():
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
                    pbar.set_description('validating...')
                    valid_acc, valid_l = self.test_(valid_iter, acc_fn, ls_fn)
                    history.add(
                        ['train_l', 'train_acc', 'valid_l', 'valid_acc'],
                        [metric[0] / metric[2], metric[1] / metric[2], valid_l, valid_acc]
                    )
            pbar.close()
        return history

    hook_mute = False

    # @staticmethod
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

    # @staticmethod
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
        # history = History('train_l', 'train_acc')
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
        # with torch.no_grad():
        #     for features, labels in test_iter:
        #         preds = self(features)
        #         test_acc = acc_func(preds, labels) / len(features)
        #         test_ls = loss(preds, labels)
        #         del preds
        #         return test_acc, test_ls.item()
        metric = Accumulator(3)
        for features, labels in test_iter:
            preds = self(features)
            metric.add(loss(preds, labels), acc_func(preds, labels), len(features))
        return metric[0] / metric[2], metric[1] / metric[2]

    @torch.no_grad()
    def predict_(self, feature_iter: Iterable,
                 unwrap_fn: Callable[[torch.Tensor], torch.Tensor] = None) -> torch.Tensor:
        ret = []
        for feature in feature_iter:
            ret.append(self(feature))
        ret = torch.cat(ret, dim=0)
        if unwrap_fn is not None:
            ret = unwrap_fn(ret)
        return ret

    @property
    def device(self):
        return self.__device

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
