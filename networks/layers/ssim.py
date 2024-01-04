import warnings

import torch
import torchvision.transforms as T
from torch import nn

from utils.func.tensor_tools import tensor_to_img, img_to_tensor

img_modes = ['L', 'RGB', '1']


def calculate_ssim(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算SSIM值
    :param y_hat: 计算对象1
    :param y: 计算对象2
    :return: 两张图片的SSIM值
    """
    mean_x, mean_y = [torch.mean(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    std_x, std_y = [torch.std(t, dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    conv_xy = torch.mean((y_hat - mean_x) * (y - mean_y), dim=list(range(1, len(y.shape))), keepdim=True)
    # 更精确的表达式
    # conv_xy =
    # torch.sum((y_hat - mean_x) * (y - mean_y), dim=list(range(1, len(y.shape))), keepdim=True)
    # /
    # (width * height - 1)
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
        computer = SSIM(self.mode, self.mute)
        return 1 - computer(y_hat, y)


class SSIM(nn.Module):

    def __init__(self, mode: str = 'L', mute=True):
        """
        SSIM计算层。计算批次中，每对y_hat与y的图片结构相似度，并求其平均作为损失值。
        计算公式为：ls =
        """
        self.mute = mute
        assert mode in img_modes, f'不支持的图像模式{mode}！'
        self.mode = mode
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        # 计算SSIM
        if self.mode == '1':
            ssim_of_each = calculate_ssim(y_hat * 255, y * 255)
        else:
            ssim_of_each = calculate_ssim(y_hat, y)
        if not self.mute:
            for ssim in ssim_of_each:
                if ssim < 0:
                    warnings.warn(f'出现了负值SSIM={ssim}！')
        return ssim_of_each
