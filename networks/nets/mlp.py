import math

import torch.nn as nn

from networks.basic_nn import BasicNN

import torch


class MLP(BasicNN):
    def __init__(self, in_features, out_features, device, base=2) -> None:
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
        layers.pop(len(layers) - 1)
        layers += [nn.Softmax(dim=0)]
        super().__init__(device, *layers)
