import torch
from torch import nn

from networks.basic_nn import BasicNN


class AlexNet(BasicNN):

    required_shape = (224, 224)

    def __init__(self, in_channels, out_features,
                 regression=False, **kwargs) -> None:
        """
        经典AlexNet模型。
        :param in_channels: 输入特征通道数。
        :param out_features: 输出通道数。
        :param regression: 是否进行回归预测。
        :param kwargs: BasicNN关键词参数。
        """
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, out_features)
        ]
        if not regression:
            layers += [nn.Softmax(dim=1)]
        super().__init__(*layers, **kwargs)
