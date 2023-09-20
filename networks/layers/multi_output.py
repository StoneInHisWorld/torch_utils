import math
import warnings
from typing import List

import numpy as np
import torch
from torch import nn

from utils import tools as tools


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
    assert in_features > 0 and out_features > 0, '输入维度与输出维度均需大于0'
    layers = [nn.Flatten(), nn.BatchNorm1d(in_features, momentum=bn_momen)] if batch_norm else [nn.Flatten()]
    layers += [nn.Linear(in_features, out_features)] if get_mlp \
        else mlp(in_features, out_features, bn_momen=bn_momen, dropout=dropout)
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if softmax:
        layers.append(nn.Softmax(dim=1))
    return layers


class DualOutputLayer(nn.Module):

    def __init__(self, in_features, fir_out, sec_out, dropout_rate=0., init_meth='normal',
                 momentum=0.) -> None:
        warnings.warn('由于双输出层与多输出层`MultiOutputLayer`功能重复，因此将在未来移除', DeprecationWarning)
        super().__init__()
        fir = nn.Sequential(
            *self.__get_layers(in_features, fir_out, dropout=dropout_rate,
                               momentum=momentum)
        )
        sec = nn.Sequential(
            *self.__get_layers(in_features, sec_out, dropout=dropout_rate,
                               momentum=momentum)
        )
        fir.apply(tools.init_wb(init_meth))
        sec.apply(tools.init_wb(init_meth))
        self.add_module('fir', fir)
        self.add_module('sec', sec)

    def forward(self, features):
        in_features_es = [
            child[1].in_features for _, child in self
        ]
        batch_size = len(features)
        feature_batch_es = [
            features.reshape((batch_size, in_fea))
            for in_fea in in_features_es
        ]
        fir_out = self[0](feature_batch_es[0])
        sec_out = self[1](feature_batch_es[1])
        return torch.hstack((fir_out, sec_out))

    def __get_layers(self, in_features: int, out_features: int, dropout=0.,
                     momentum=0.) -> List[nn.Module]:
        assert in_features > 0 and out_features > 0, '输入维度与输出维度均需大于0'
        # layers = [nn.BatchNorm1d(in_features)]
        # # 构造一个三层感知机
        # trans_layer_sizes = [in_features, (in_features + out_features) // 2, out_features]
        # # 对于每个layer_size加入全连接层、BN层以及Dropout层
        # for i in range(len(trans_layer_sizes) - 1):
        #     in_size, out_size = trans_layer_sizes[i], trans_layer_sizes[i + 1]
        #     layers += [
        #         nn.Linear(in_size, out_size),
        #         # nn.BatchNorm1d(out_size, momentum=momentum),
        #         nn.LeakyReLU()
        #     ]
        #     if dropout > 0.:
        #         layers.append(nn.Dropout())
        # # 去掉最后的Dropout层
        # if type(layers[-1]) == nn.Dropout:
        #     layers.pop(-1)
        # # 将最后的激活函数换成Softmax
        # layers.pop(-1)
        # 加入Softmax层
        layers = [
            nn.Flatten(),
            nn.Linear(in_features, out_features)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Softmax(dim=1))
        return layers

    def __iter__(self):
        return self.named_children()

    def __getitem__(self, item: int):
        children = self.named_children()
        for _ in range(item):
            next(children)
        return next(children)[1]


class MultiOutputLayer(nn.Module):

    def __init__(self, in_features, out_or_strategy: List,
                 init_meth='normal', self_defined=False,
                 dropout_rate=0., momentum=0.,
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
            p.apply(tools.init_wb(init_meth))
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
