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

    def __init__(self, reduction='mean'):
        """PCC损失层。计算每对y_hat与y的皮尔逊相关系数，并求其平均逆作为损失值。"""
        super().__init__()
        self.computer = PCC(reduction)

    def forward(self, y_hat, y):
        return 1 - self.computer(y_hat, y)


class PCC(nn.Module):

    def __init__(self, reduction='mean'):
        r"""PCC计算层。计算批次中每对y_hat与y的皮尔逊相关系数，通过张量的形式返回。

        计算PCC值，公式为：
        .. math::
            \mathrm{pcc}(\hat{y}, y) = cov(\hat{y}, y) / ( \mathrm{\sigma_{\hat{y}}} \mathrm{\sigma_{y}})
        """
        self.reduction = reduction
        supported = ['mean', 'sum', 'none']
        assert reduction in supported, f'不支持的reduction方式{reduction}，支持的包括{supported}'
        super().__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        result = calculate_pcc(y_hat, y).squeeze()
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result
        # if self.size_averaged:
        #     return result.mean()
        # else:
        #     return result.squeeze()
