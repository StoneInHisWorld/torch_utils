import torch
from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.multi_output import MultiOutputLayer


class Pix2Pix(BasicNN):
    required_shape = (256, 256)

    def __init__(self, input_channel, out_features, base_channel=64,
                 kernel_size=4, bn_momen=0.8, init_meth='normal', with_checkpoint=False, device='cpu'):
        cp_layer = lambda i, o: nn.Sequential(
            nn.Conv2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.ReLU()
        )
        ep_layer = lambda i, o: nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(i, o, kernel_size=kernel_size, stride=1, padding=2),
            nn.ReLU()
        )
        self.contracting_path = [
            cp_layer(input_channel, base_channel),
            cp_layer(base_channel, base_channel * 2),
            cp_layer(base_channel * 2, base_channel * 4),
            cp_layer(base_channel * 4, base_channel * 8),
            cp_layer(base_channel * 8, base_channel * 8),
            cp_layer(base_channel * 8, base_channel * 8),
        ]
        self.expanding_path = [
            ep_layer(base_channel * 8, base_channel * 8),
            ep_layer(base_channel * 16, base_channel * 8),
            ep_layer(base_channel * 16, base_channel * 8),
            ep_layer(base_channel * 16, base_channel * 8),
            ep_layer(base_channel * 8, base_channel * 2),
            ep_layer(base_channel * 4, base_channel)
        ]
        self.output_path = [
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=kernel_size, stride=1, padding=1),
            nn.Conv2d(base_channel * 2, out_features[0], kernel_size=kernel_size, stride=1, padding=1),
        ]
        super().__init__(device, init_meth, with_checkpoint, *self.contracting_path, *self.expanding_path,
                         *self.output_path)

    def forward(self, input):
        cp_results = []
        for layer in self.contracting_path:
            input = layer(input)
            cp_results.append(input)
        for i, layer in enumerate(self.expanding_path):
            input = layer(input)
            input = torch.hstack((input, cp_results[i]))
        return input

