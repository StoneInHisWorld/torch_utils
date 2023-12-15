import math
from collections.abc import Iterable
from typing import List

import numpy as np
import torch
from torch import nn

import utils.func.torch_tools
from utils.func import pytools as tools


def mlp(in_features, out_features, base=2, bn_momen=0., dropout=0.) -> List[nn.Module]:
    layers = []
    # 计算过渡层维度
    trans_layer_sizes = np.logspace(
        math.log(in_features, base),
        math.log(out_features, base),
        int(math.log(abs(in_features - out_features), base)),
        base=base
    )
    trans_layer_sizes = list(map(int, trans_layer_sizes))
    # 保证头尾两元素符合输入输出
    trans_layer_sizes = [in_features, *trans_layer_sizes[1: -1], out_features] \
        if len(trans_layer_sizes) > 2 else [in_features, out_features]
    # 对于每个layer_size加入全连接层、BN层以及Dropout层
    for i in range(len(trans_layer_sizes) - 1):
        in_size, out_size = trans_layer_sizes[i], trans_layer_sizes[i + 1]
        layers.append(nn.Linear(in_size, out_size))
        layers.append(
            nn.BatchNorm1d(out_size, momentum=bn_momen)
        )
        layers.append(nn.LeakyReLU())
        if dropout > 0.:
            layers.append(nn.Dropout())
    return layers


def linear_output(in_features: int, out_features: int,
                  softmax=True, batch_norm=True, get_mlp=False,
                  dropout=0., bn_momen=0.) -> List[nn.Module]:
    """
    构造一个线性输出通道。将输入数据展平，利用多层线性层，提取特征，输出指定通道数目的数据。
    :param in_features: 输入数据特征通道数
    :param out_features: 输出数据特征通道数
    :param softmax: 是否使用softmax层，若为true，则在通道最后添加nn.Softmax()
    :param batch_norm: 是否使用BatchNorm层，若为true，则在通道最后添加nn.BatchNorm1d()
    :param get_mlp: 是否使用更复杂的多层感知机，若为true，则本输出通道的构造方法将更换为复杂的感知机。
    :param dropout: 是否使用dropout层，若为所填数值>0，则在通道最后添加nn.Dropout(dropout)（位于softmax层之前）
    :param bn_momen: BatchNorm层的动量参数，仅在batch_norm为true时有效
    :return: 构造的输出通道
    """
    assert in_features > 0 and out_features > 0, '输入维度与输出维度均需大于0'
    layers = [nn.Flatten(), nn.BatchNorm1d(in_features, momentum=bn_momen)] if batch_norm else [nn.Flatten()]
    layers += [nn.Linear(in_features, out_features)] if not get_mlp \
        else mlp(in_features, out_features, bn_momen=bn_momen, dropout=dropout)
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if softmax:
        layers.append(nn.Softmax(dim=1))
    return layers


class MultiOutputLayer(nn.Module):

    def __init__(self, in_features, out_or_strategy: Iterable, self_defined=False,
                 init_meth='normal', dropout_rate=0., momentum=0.
                 ) -> None:
        """
        多通道输出层。将单通道输入扩展为多通道输出。
        :param in_features: 输入特征列数。
        :param init_meth: 线性层或卷积层初始化方法。
        :param self_defined: 是否使用自定义通道。若为True，则需要用户通过out_or_strategy自定义路径结构
        :param dropout_rate: Dropout层比例
        :param momentum: BatchNorm层动量超参数
        :param out_or_strategy: 若self_defined为False，则该项为输出特征列数，列数的数量对应输出路径数
        """
        super().__init__()
        self.in_features = in_features
        self._paths = [
            nn.Sequential(*linear_output(in_features, o, dropout=dropout_rate, bn_momen=momentum))
            for o in out_or_strategy
        ] if not self_defined else [
            nn.Sequential(*s)
            for s in out_or_strategy
        ]
        for i, p in enumerate(self._paths):
            p.apply(utils.func.torch_tools.init_wb(init_meth))
            self.add_module(f'path{i}', p)

    def forward(self, features):
        outs = [m(features) for _, m in self]
        return torch.hstack(outs)

    def __iter__(self):
        return self.named_children()

    def __getitem__(self, item: int):
        children = self.named_children()
        for _ in range(item):
            next(children)
        return next(children)[1]  # next()得到的是（名字，模块）
