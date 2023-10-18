import torch.nn as nn

from networks.basic_nn import BasicNN


class LeNet(BasicNN):

    required_shape = (32, 32)

    def __init__(self, in_channels, out_features, device='cpu', init_meth='normal',
                 with_checkpoint=False, regression=False) -> None:
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, out_features)
        ]
        if not regression:
            layers += [nn.Softmax(dim=1)]
        super().__init__(device, init_meth, with_checkpoint, *layers)
