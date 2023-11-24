import torch
from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.common_layers import Val2Fig


class Pix2Pix(BasicNN):
    required_shape = (256, 256)

    def __init__(self, input_channel, out_features,
                 base_channel=4, kernel_size=4, bn_momen=0.8,
                 init_meth='normal', with_checkpoint=False, device='cpu',
                 output_img=None):
        """
        摘录于王志远硕士毕业论文。
        :param input_channel: 输入数据通道，一般是图片通道数。
        :param out_features: 输出特征维度，需包含三个元素，分别为（图片通道数，图片长度，图片宽度）
        :param base_channel: 决定网络复杂度的基础通道数，需为大于0的整数。数值越高决定提取的特征维度越高。
        :param kernel_size: 卷积层使用的感受野大小
        :param bn_momen: 批量标准化层的动量超参数
        :param init_meth:
        :param with_checkpoint:
        :param device:
        """
        cp_layer = lambda i, o: nn.Sequential(
            nn.Conv2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.ReLU()
        )
        ep_layer = lambda i, o: nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(i, o, kernel_size=kernel_size + 1, stride=1, padding=2),
            nn.ReLU()
        )
        self.contracting_path = [
            cp_layer(input_channel, base_channel),
            cp_layer(base_channel, base_channel * 2),
            cp_layer(base_channel * 2, base_channel * 4),
            cp_layer(base_channel * 4, base_channel * 8),
            cp_layer(base_channel * 8, base_channel * 8),
            cp_layer(base_channel * 8, base_channel * 8),
            cp_layer(base_channel * 8, base_channel * 8),
        ]
        self.expanding_path = [
            ep_layer(base_channel * 8, base_channel * 8),
            ep_layer(base_channel * 16, base_channel * 8),
            ep_layer(base_channel * 16, base_channel * 8),
            ep_layer(base_channel * 16, base_channel * 4),
            ep_layer(base_channel * 8, base_channel * 2),
            ep_layer(base_channel * 4, base_channel),
        ]
        self.output_path = [
            ep_layer(base_channel * 2, base_channel * 2),
            nn.Conv2d(base_channel * 2, out_features[0], kernel_size=kernel_size + 1, stride=1, padding=2),
        ]
        if output_img is not None:
            self.output_path.append(Val2Fig(output_img))
        super().__init__(device, init_meth, with_checkpoint, *self.contracting_path, *self.expanding_path,
                         *self.output_path)

    def forward(self, input):
        cp_results = []
        for layer in self.contracting_path:
            input = layer(input)
            cp_results.append(input)
        cp_results = reversed(cp_results[:-1])  # 需要去除掉最后一个结果
        for layer in self.expanding_path:
            input = layer(input)
            input = torch.hstack((input, next(cp_results)))
        for layer in self.output_path:
            input = layer(input)
        return input

