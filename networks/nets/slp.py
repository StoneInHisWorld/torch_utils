import torch.nn as nn

from networks.basic_nn import BasicNN


class SLP(BasicNN):

    def __init__(self, in_features, out_features, regression=True,
                 **kwargs) -> None:
        """
        经典单层感知机。
        :param in_channels: 输入特征通道数。
        :param out_features: 输出数据通道数。
        :param regression: 是否进行回归预测。
        :param kwargs: BasicNN关键词参数。
        """
        layers = [
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout()
        ]
        if not regression:
            layers += [nn.Softmax(dim=1)]
        super().__init__(*layers, **kwargs)


