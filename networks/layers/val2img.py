import torch
from torch import nn

from data_related import data_related as dr


class Val2Fig(nn.Module):

    def __init__(self, img_mode='L'):
        """
        数值-图片转化层。根据指定模式，对数值进行归一化后反归一化为图片模式像素取值范围，从而转化为可视图片。
        :param img_mode: 生成的图片模式。'L'为灰度图，‘RGB’为彩色图。
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
