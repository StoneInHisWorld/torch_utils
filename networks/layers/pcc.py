import torch
from torch import nn

img_modes = ['L', '1']


def calculate_pcc(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算PCC值，公式为：
    .. math::
        \mathrm{pcc}(\hat{y}, y) = cov(\hat{y}, y) / ( \mathrm{\sigma_{\hat{y}}} \mathrm{\sigma_{y}})
    :param y_hat: 计算对象1
    :param y: 计算对象2
    :return: 两张图片的PCC值
    """
    mean_x, mean_y = [t.mean(dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    std_x, std_y = [t.std(dim=list(range(1, len(t.shape))), keepdim=True) for t in [y_hat, y]]
    conv_xy = ((y_hat - mean_x) * (y - mean_y)).sum(dim=list(range(1, len(y.shape))), keepdim=True) / (
            y.shape[-1] * y.shape[-2] - 1
    )
    denominator = std_x * std_y
    return conv_xy / denominator


class PCCLoss(nn.Module):

    def __init__(self, size_averaged=True):
        """PCC损失层。计算每对y_hat与y的皮尔逊相关系数，并求其平均逆作为损失值。"""
        self.size_averaged = size_averaged
        super().__init__()

    def forward(self, y_hat, y):
        return 1 - PCC(self.size_averaged)(y_hat, y)


class PCC(nn.Module):

    def __init__(self, size_averaged=True):
        r"""PCC计算层。计算批次中每对y_hat与y的皮尔逊相关系数，通过张量的形式返回。

        计算PCC值，公式为：
        .. math::
            \mathrm{pcc}(\hat{y}, y) = cov(\hat{y}, y) / ( \mathrm{\sigma_{\hat{y}}} \mathrm{\sigma_{y}})
        """
        self.size_averaged = size_averaged
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        result = calculate_pcc(y_hat, y)
        if self.size_averaged:
            return result.mean()
        else:
            return result.squeeze()
