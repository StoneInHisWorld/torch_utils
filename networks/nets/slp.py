import torch.nn as nn

from networks.basic_nn import BasicNN


class SLP(BasicNN):
    def __init__(self, in_features, out_features, device) -> None:
        layers = [
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Softmax(dim=1)
        ]
        super().__init__(device, *layers)


