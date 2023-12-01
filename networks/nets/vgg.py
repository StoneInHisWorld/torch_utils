from collections.abc import Iterable
from typing import Tuple

from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.multi_output import MultiOutputLayer, linear_output

VGG_11 = (
    (1, 64), (1, 128), (2, 256), (2, 512), (2, 512)
)


class VGG(BasicNN):

    required_shape = (224, 224)

    def __init__(self, in_channels: int, out_features: Iterable or int,
                 conv_arch: Tuple[int, int] = VGG_11, regression=False,
                 **kwargs) -> None:
        """
        经典VGG网络模型，可通过指定conv_arch构造指定版本的VGG网络。
        :param in_channels: 输入特征通道数。
        :param out_features: 输出数据通道数。
        :param regression: 是否进行回归预测。
        :param kwargs: BasicNN关键词参数。
        :param conv_arch: 需要构造的VGG版本，可通过vgg.py内获取对应版本的结构。
        """
        conv_blks = [
            nn.BatchNorm2d(in_channels)
        ]
        for (num_convs, out_channels) in conv_arch:
            conv_blks += [
                VGGBlock(num_convs, in_channels, out_channels),
            ]
            in_channels = out_channels
        conv_blks += [
            nn.Flatten(),
            nn.BatchNorm1d(in_channels * 7 * 7),
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            MultiOutputLayer(1024, out_features, init_meth=kwargs['init_meth']) if isinstance(out_features, Iterable)
            else nn.Sequential(*linear_output(1024, out_features, softmax=not regression))
        ]
        super().__init__(*conv_blks, **kwargs)


class VGGBlock(nn.Sequential):

    def __init__(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3,
                    padding=1
                ),
                nn.ReLU()
            ]
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        super().__init__(*layers)
