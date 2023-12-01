import math

import torch.nn as nn

from networks.basic_nn import BasicNN

import torch


class MLP(BasicNN):

    def __init__(self, in_features, out_features,
                 base=2, regression=True, **kwargs) -> None:
        """
        经典多层感知机。
        :param in_channels: 输入特征通道数。
        :param out_features: 输出数据通道数。
        :param regression: 是否进行回归预测。
        :param kwargs: BasicNN关键词参数。
        :param base: 多层感知机复杂度参数。base要求为正整数，越大意味着多层感知机越复杂。
        """
        layer_sizes = torch.logspace(
            math.log(in_features, base),
            math.log(out_features, base),
            int(math.log(abs(in_features - out_features), base)),
            base=base
        )
        layer_sizes = list(map(int, layer_sizes))
        # 保证头尾两元素符合输入输出
        layer_sizes = [in_features, *layer_sizes[1: -1], out_features] \
            if len(layer_sizes) > 2 else [in_features, out_features]
        layers = [
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
        ]
        # 对于每个layer_size加入全连接层、BN层以及Dropout层
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            layers += [
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(),
            ]
        # layers.pop(len(layers) - 1)
        if not regression:
            layers += [nn.Softmax(dim=1)]
        super().__init__(*layers, **kwargs)
