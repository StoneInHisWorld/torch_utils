import warnings

import torch
from torch import nn
import utils.tools as tools
import torchvision.transforms as T


def calculate_ssim(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x, mean_y = [torch.mean(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    std_x, std_y = [torch.std(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    conv_xy = torch.sum((y_hat - mean_x) * (y - mean_y), dim=list(range(1, len(y.shape))), keepdim=True) / (len(y_hat) - 1)
    R = torch.tensor(255)
    c1, c2 = torch.sqrt(R * 0.01), torch.sqrt(R * 0.03)
    numerator = (2 * mean_x * mean_y + c1) * (2 * conv_xy + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2)
    return numerator / denominator


class SSIMLoss(nn.Module):

    def __init__(self, mode: str = 'L', mute=True):
        """
        SSIM损失层。计算每对y_hat与y的图片结构相似度，并求其平均逆（1 - mean(ssim)）作为损失值。
        计算公式为：ls =
        """
        self.mute = mute
        self.mode = mode
        super().__init__()

    def forward(self, y_hat, y):
        # 将y_hat, y反归一化
        # TODO: Untested!
        y_hat, y = [tools.tensor_to_img(t, self.mode) for t in [y_hat, y]]
        transformer = T.PILToTensor()
        y_hat, y = [transformer(t) for t in [y_hat, y]]
        # 计算SSIM
        ssim_of_each = calculate_ssim(y_hat, y)
        if not self.mute:
            for ssim in ssim_of_each:
                if ssim < 0:
                    warnings.warn(f'出现了负值SSIM={ssim}！')
        return 1 - torch.mean(ssim_of_each, dim=list(range(1, len(ssim_of_each.shape))))


class SSIM(nn.Module):

    def __init__(self, mode: str = 'L', mute=True):
        """
        SSIM计算层。计算批次中，每对y_hat与y的图片结构相似度，并求其平均作为损失值。
        计算公式为：ls =
        """
        self.mute = mute
        self.mode = mode
        super().__init__()

    def forward(self, y_hat, y):
        # 将y_hat, y反归一化
        # TODO: Untested!
        y_hat, y = [tools.tensor_to_img(t, self.mode) for t in [y_hat, y]]
        transformer = T.PILToTensor()
        y_hat, y = [transformer(t) for t in [y_hat, y]]
        # 计算SSIM
        ssim_of_each = calculate_ssim(y_hat, y)
        if not self.mute:
            for ssim in ssim_of_each:
                if ssim < 0:
                    warnings.warn(f'出现了负值SSIM={ssim}！')
        return torch.mean(ssim_of_each, dim=list(range(1, len(ssim_of_each.shape))))
