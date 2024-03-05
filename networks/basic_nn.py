import os.path
import warnings
from typing import Callable

import torch
import torch.nn as nn
from networks.trainer import Trainer
from torch.nn import Module
from torch.utils import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.func.torch_tools as ttools
from data_related.dataloader import LazyDataLoader
from utils.accumulator import Accumulator


class BasicNN(nn.Sequential):
    required_shape = (-1,)

    def __init__(self, *args: Module, **kwargs) -> None:
        """基本神经网络。
        提供神经网络的基本功能，包括权重初始化，训练以及测试。
        :param args: 需要添加的网络层
        :param kwargs: 可选提供的参数。
        -- device: 网络所处设备。
        -- init_meth: 网络初始化方法。
        -- init_args: 网络初始化方法所用参数。
        -- with_checkpoint: 是否使用检查点机制。
        """
        # 设置默认值
        init_meth = 'zero' if 'init_meth' not in kwargs.keys() else kwargs['init_meth']
        device = torch.device('cpu') if 'device' not in kwargs.keys() else kwargs['device']
        with_checkpoint = False if 'with_checkpoint' not in kwargs.keys() else kwargs['with_checkpoint']
        init_args = [] if 'init_args' not in kwargs.keys() else kwargs['init_args']
        # 初始化各模块
        super().__init__(*args)
        self.__init_submodules(init_meth, *init_args)
        self.apply(lambda m: m.to(device))

        self.__device = device
        if with_checkpoint:
            warnings.warn('使用“检查点机制”虽然会减少前向传播的内存使用，但是会大大增加反向传播的计算量！')
        self.__checkpoint = with_checkpoint

    def train_(self,
               data_iter, optimizer, acc_fn,
               n_epochs=10, ls_fn: nn.Module = nn.L1Loss(), lr_scheduler=None,
               valid_iter=None, k=1, n_workers=1, hook=None
               ):
        """
        神经网络训练函数，调用Trainer进行训练。
        :param data_iter: 训练数据供给迭代器。
        :param optimizer: 网络参数优化器。
        :param acc_fn: 准确率计算函数。签名需为：acc_fn(predict: tensor, labels: tensor, size_averaged: bool = True) -> ls: tensor or float
        :param n_epochs: 迭代世代数。
        :param ls_fn: 训练损失函数。签名需为：ls_fn(predict: tensor, labels: tensor) -> ls: tensor，其中不允许有销毁梯度的操作。
        :param lr_scheduler: 学习率规划器，用于动态改变学习率。若不指定，则会使用固定学习率规划器。
        :param valid_iter: 验证数据供给迭代器。
        :param hook: 是否使用hook机制跟踪梯度变化。可选填入[None, 'mute', 'full']
        :param n_workers: 进行训练的处理机数量。
        :param k: 进行训练的k折数，指定为1则不进行k-折训练，否则进行k-折训练，并且data_iter为k-折训练数据供给器，valid_iter会被忽略.
        :return: 训练数据记录`History`对象
        """
        if hook is None:
            with_hook, hook_mute = False, False
        elif hook == 'mute':
            with_hook, hook_mute = True, True
        else:
            with_hook, hook_mute = True, False
        # TODO: optimizer以及lr_scheduler需要支持多个
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
        with Trainer(self,
                     data_iter, optimizer, lr_scheduler, acc_fn, n_epochs, ls_fn,
                     with_hook, hook_mute) as trainer:
            if k > 1:
                # 是否进行k-折训练
                return trainer.train_with_k_fold(data_iter, k, n_workers)
            elif n_workers <= 1:
                # 判断是否要多线程
                if valid_iter:
                    return trainer.train_and_valid(valid_iter)
                else:
                    return trainer.train()
            else:
                return trainer.train_with_threads(valid_iter)

    @torch.no_grad()
    def test_(self, test_iter,
              acc_fn: Callable[[torch.Tensor, torch.Tensor], float or torch.Tensor],
              ls_fn: Callable[[torch.Tensor, torch.Tensor], float or torch.Tensor] = nn.L1Loss,
              ) -> [float, float]:
        """测试方法。
        取出迭代器中的下一batch数据，进行预测后计算准确度和损失
        :param test_iter: 测试数据迭代器
        :param acc_fn: 计算准确度所使用的函数，该函数需要求出整个batch的准确率之和。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param ls_fn: 计算损失所使用的函数，该函数需要求出整个batch的损失平均值。签名需为：loss(Y_HAT, Y) -> float or torch.Tensor
        :return: 测试准确率，测试损失
        """
        self.eval()
        metric = Accumulator(3)
        for features, labels in test_iter:
            preds = self(features)
            metric.add(acc_fn(preds, labels), ls_fn(preds, labels) * len(features), len(features))
        return metric[0] / metric[2], metric[1] / metric[2]

    @torch.no_grad()
    def predict_(self, data_iter: DataLoader or LazyDataLoader,
                 acc_fn: Callable,
                 ls_fn: Callable[[torch.Tensor, torch.Tensor], float or torch.Tensor] = nn.L1Loss,
                 unwrap_fn: Callable[
                     [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None
                 ) -> torch.Tensor:
        """预测方法。
        对于数据迭代器中的每一batch数据，保存输入数据、预测数据、标签集、准确率、损失值数据，并打包返回。
        :param data_iter: 预测数据迭代器。
        :param acc_fn: 计算准确度所使用的函数，该函数需要求出整个batch的准确率之和。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param ls_fn: 计算损失所使用的函数，该函数需要求出整个batch的损失平均值。签名需为：loss(Y_HAT, Y) -> float or torch.Tensor
        :param unwrap_fn: 对所有数据进行打包的方法。如不指定，则直接返回预测数据。签名需为：unwrap_fn(inputs, predictions, labels, acc_s, loss_es) -> Any
        :return: 打包好的数据集
        """
        self.eval()
        if unwrap_fn is not None:
            # 将本次预测所产生的全部数据打包并返回。
            inputs, predictions, labels, acc_s, loss_es = [], [], [], [], []
            with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
                # 对每个批次进行预测，并进行acc和loss的计算
                for fe_batch, lb_batch in data_iter:
                    inputs.append(fe_batch)
                    pre_batch = self(fe_batch)
                    predictions.append(pre_batch)
                    labels.append(lb_batch)
                    # # TODO：对单独的每张图片进行acc和loss计算。是否过于效率低下？
                    # for pre, lb in zip(pre_batch, lb_batch):
                    #     acc_s.append(
                    #         acc_fn(pre.reshape(1, *pre.shape), lb.reshape(1, *pre.shape))
                    #     )
                    #     loss_es.append(ls_fn(pre, lb))
                    acc_s += acc_fn(pre_batch, lb_batch, size_average=False)
                    keeping_dims = list(range(len(fe_batch.shape)))[1:]
                    loss_es += ls_fn(pre_batch, lb_batch).mean(dim=keeping_dims)
                data_iter.set_description('正对结果进行解包……')
            inputs = torch.cat(inputs, dim=0)
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)
            acc_s = torch.tensor(acc_s)
            loss_es = torch.tensor(loss_es)
            predictions = unwrap_fn(inputs, predictions, labels, acc_s, loss_es)
        else:
            # 如果不需要打包数据，则直接返回预测数据集
            predictions = []
            with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
                for fe_batch, lb_batch in data_iter:
                    predictions.append(self(fe_batch))
            predictions = torch.cat(predictions, dim=0)
        return predictions

    @property
    def device(self):
        return self.__device

    def load_state_dict_(self, where: str):
        assert os.path.exists(where), f'目录{where}无效！'
        assert where.endswith('.ptsd'), f'该文件{where}并非网络参数文件！'
        paras = torch.load(where)
        self.load_state_dict(paras)

    def __init_submodules(self, init_meth, *args):
        init_meth = ttools.init_wb(init_meth)
        if init_meth is not None:
            self.apply(init_meth)
        else:
            try:
                where = args[0]
                if where.endswith('.ptsd'):
                    paras = torch.load(where)
                    self.load_state_dict(paras)
                elif where.endswith('.ptm'):
                    raise NotImplementedError('针对预训练好的网络，请使用如下方法获取`net = torch.load("../xx.ptm")`')
            except IndexError:
                raise ValueError('选择预训练好的参数初始化网络，需要在初始化方法的第一参数提供参数或者模型的路径！')
            except FileNotFoundError:
                raise FileNotFoundError(f'找不到{where}！')

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
