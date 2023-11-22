import torch
from torch import nn


def SSIM(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x, mean_y = [torch.mean(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    var_x, var_y = [torch.var(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    conv_xy = torch.mean((y_hat - mean_x) * (y - mean_y))
    R = torch.tensor(255)
    c1, c2 = torch.sqrt(R * 0.01), torch.sqrt(R * 0.03)
    numerator = (2 * mean_x * mean_y + c1) * (2 * conv_xy + c2)
    denominator = (mean_x ** 2 * mean_y ** 2 + c1) * (var_x ** 2 * var_y ** 2 + c2)
    return numerator / denominator


class SSIMLoss(nn.Module):

    def forward(self, y_hat, y):
        ssim_of_each_sample = SSIM(y_hat, y)
        return torch.mean(1 - ssim_of_each_sample)
