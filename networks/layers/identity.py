from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x