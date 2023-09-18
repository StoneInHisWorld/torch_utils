from torch import nn

import networks.layers.common_layers as cl
import networks.layers.multi_output
from networks.basic_nn import BasicNN


class WZYNetEssay(BasicNN):

    required_shape = (256, 256)

    def __init__(self, input_channel, base_channel, out_features, device='cpu', kernel_size=3, bn_momen=0.95,
                 init_meth='normal', with_checkpoint=False):
        layers = [
            nn.BatchNorm2d(input_channel, momentum=bn_momen),
            nn.Conv2d(input_channel, base_channel, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channel, base_channel, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channel, momentum=bn_momen),
            nn.Conv2d(base_channel, base_channel * 2, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channel * 2, momentum=bn_momen), nn.Dropout(0.3),
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channel * 2, momentum=bn_momen),
            nn.Conv2d(base_channel * 2, base_channel * 3, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channel * 3, momentum=bn_momen), nn.Dropout(0.3),
            nn.Conv2d(base_channel * 3, base_channel * 3, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channel * 3, momentum=bn_momen),
            nn.Conv2d(base_channel * 3, base_channel * 4, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channel * 4, momentum=bn_momen), nn.MaxPool2d(2),
            networks.layers.multi_output.DualOutputLayer(
                base_channel * 4, out_features[0], out_features[1],
                momentum=bn_momen,
            ),
        ]
        super().__init__(device, init_meth, with_checkpoint, *layers)
