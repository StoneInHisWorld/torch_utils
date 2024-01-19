import torch
from torch import nn

img_modes = ['L', 'RGB', '1']


def calculate_ssim(y_hat: torch.Tensor, y: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    计算SSIM值
    :param y_hat: 计算对象1
    :param y: 计算对象2
    :param R: 像素范围
    :return: 两张图片的SSIM值
    """
    mean_x, mean_y = [t.mean(dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    var_x, var_y = [t.var(dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    conv_xy = ((y_hat - mean_x) * (y - mean_y)).sum(dim=list(range(1, len(y.shape))), keepdim=True) / (
            y.shape[-1] * y.shape[-2] - 1
    )
    c1, c2 = torch.sqrt(R * 0.01), torch.sqrt(R * 0.03)
    numerator = (2 * mean_x * mean_y + c1) * (2 * conv_xy + c2)
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)
    return numerator / denominator


class SSIMLoss(nn.Module):

    def __init__(self, mode: str = 'L', size_average=True):
        """
        SSIM损失层。计算每对y_hat与y的图片结构相似度，并求其平均逆（1 - mean(ssim)）作为损失值。
        计算公式为：ls =
        """
        self.mode = mode
        self.size_average = size_average
        super().__init__()

    def forward(self, y_hat, y):
        computer = SSIM(self.mode)
        if self.size_average:
            return torch.mean(1 - computer(y_hat, y))
        else:
            return 1 - computer(y_hat, y)


class SSIM(nn.Module):

    def __init__(self, mode: str = 'L'):
        """
        SSIM计算层。计算批次中，每对y_hat与y的图片结构相似度，并求其平均作为损失值。
        计算公式为：ls =
        """
        assert mode in img_modes, f'不支持的图像模式{mode}！'
        self.mode = mode
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        # 计算SSIM
        if self.mode == '1':
            R = torch.tensor(1, dtype=y_hat.dtype, device=y_hat.device)
        else:
            R = torch.tensor(255, dtype=y_hat.dtype, device=y_hat.device)
        return calculate_ssim(y_hat, y, R)
