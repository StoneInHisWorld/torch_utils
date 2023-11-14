from collections.abc import Iterable
from typing import Tuple

import torch
from torch import nn

from networks.layers.multi_output import MultiOutputLayer, linear_output
from networks.basic_nn import BasicNN

VGG_11 = (
    (1, 64), (1, 128), (2, 256), (2, 512), (2, 512)
)


class VGG(BasicNN):

    required_shape = (224, 224)

    # TODO: not suitable for current dataset

    def __init__(self, in_channels: int, out_features: Iterable | int,
                 conv_arch: Tuple[int, int] = VGG_11, init_meth='xavier',
                 with_checkpoint=False, device: torch.device = 'cpu', regression=False) -> None:
        conv_blks = [
            nn.BatchNorm2d(in_channels)
        ]
        for (num_convs, out_channels) in conv_arch:
            conv_blks += [
                VGGBlock(num_convs, in_channels, out_channels),
            ]
            in_channels = out_channels
        # 适用于Vortex
        conv_blks += [
            nn.Flatten(),
            # nn.BatchNorm1d(in_channels * 7 * 7),
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            # networks.layers.multi_output.DualOutputLayer(4096, out_feature[0], out_feature[1],
            #                                              dropout_rate=0.5, momentum=0.95)
            MultiOutputLayer(1024, out_features, init_meth=init_meth) if isinstance(out_features, Iterable)
            else nn.Sequential(*linear_output(1024, out_features, softmax=not regression))
        ]
        super().__init__(device, init_meth, with_checkpoint, *conv_blks)


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