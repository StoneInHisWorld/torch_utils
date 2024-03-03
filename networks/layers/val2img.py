import torch
from torch import nn


class Val2Fig(nn.Module):

    supported = ['RGB', 'L', '1']

    def __init__(self, img_mode='L'):
        """数值-图片转化层。
        根据指定模式，对数值进行归一化后转化为图片模式像素取值范围。计算过程中没有使用非数学计算函数，理论上不会干扰求导。

        公式为：
        .. math::
            \mathrm{rangefloor_pixel} = min(\mathrm{range_pixel})

            \mathrm{figure_valued}(t) =
            (t - t_min) * \mathrm{range_pixel}
            /
            (t_max - t_min)
            + \mathrm{rangefloor_pixel}

        经测试，转化后的数值图片会有失真，但可以正常计算loss值。

        :param img_mode: 生成的图片模式。'L'为灰度图，‘RGB’为彩色图, '1'为二值黑白图。
        """
        assert img_mode in Val2Fig.supported, f'不支持的图片模式{img_mode}！支持的图片模式包括{Val2Fig.supported}'
        self.mode = img_mode
        super().__init__()

    def forward(self, input: torch.Tensor):
        # 进行归一化
        # input = dr.normalize(input)
        input_min = input.min(
            dim=len(input.shape) - 1, keepdim=True
        )[0].min(
            dim=len(input.shape) - 2, keepdim=True
        )[0]
        input_max = input.max(
            dim=len(input.shape) - 1, keepdim=True
        )[0].max(
            dim=len(input.shape) - 2, keepdim=True
        )[0]
        input_range = input_max - input_min
        if self.mode == 'L' or self.mode == 'RGB':
            expected_range, floor = 255, 0
        elif self.mode == '1':
            expected_range, floor = 1, 0
        else:
            expected_range, floor = input_range, input_min
        float_result = ((input - input_min)/input_range) * expected_range + floor
        return float_result
