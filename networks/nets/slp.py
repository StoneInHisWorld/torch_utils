import torch.nn as nn

from networks.basic_nn import BasicNN


class SLP(BasicNN):

    def __init__(self, in_features, out_features, device='cpu', init_meth='normal',
                 with_checkpoint=False, regression=True) -> None:
        layers = [
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout()
        ]
        if not regression:
            layers += [nn.Softmax(dim=1)]
        super().__init__(device, init_meth, with_checkpoint, *layers)


