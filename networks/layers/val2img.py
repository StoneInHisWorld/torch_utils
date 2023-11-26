import torch
from torch import nn

from utils import data_related as dr


class Val2Fig(nn.Module):

    def __init__(self, img_mode='L'):
        """
        将数值转化为要求模式的图片。
        :param img_mode: 生成的图片模式。'L'为灰度图。
        """
        self.mode = img_mode
        super().__init__()

    def forward(self, y_hat: torch.Tensor):
        # 进行归一化
        y_hat = dr.normalize(y_hat)
        if self.mode == 'L' or self.mode == 'RGB':
            return (y_hat + 1) * 128
        else:
            return y_hat
