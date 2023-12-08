import warnings

import torch
from torch import nn


def SSIM(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # TODO：conv_xy大得吓人！
    mean_x, mean_y = [torch.mean(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    std_x, std_y = [torch.std(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    conv_xy = torch.sum((y_hat - mean_x) * (y - mean_y), dim=list(range(1, len(y.shape))), keepdim=True) / (len(y_hat) - 1)
    R = torch.tensor(255)
    c1, c2 = torch.sqrt(R * 0.01), torch.sqrt(R * 0.03)
    numerator = (2 * mean_x * mean_y + c1) * (2 * conv_xy + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2)
    return numerator / denominator


class SSIMLoss(nn.Module):

    def __init__(self, mute=True):
        """
        SSIM损失层。计算每对y_hat与y的图片结构相似度，并求其平均逆作为损失值。
        计算公式为：ls =
        """
        self.mute = mute
        super().__init__()

    def forward(self, y_hat, y):
        ssim_of_each_sample = SSIM(y_hat, y)
        for ssim in ssim_of_each_sample:
            if ssim < 0 and not self.mute:
                warnings.warn(f'出现了负值SSIM={ssim}！')
        return torch.mean(1 - ssim_of_each_sample)
