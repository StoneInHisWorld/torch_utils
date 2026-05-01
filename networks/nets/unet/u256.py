import torch
from torch import nn
from ...basic_nn import BasicNN


class UNet256(BasicNN):

    def __init__(self, input_channel, out_channel,
                 base_channel=64, kernel_size=4, bn_momen=0.8, dropout=0.,
                 **kwargs):
        """
        适用于图片翻译、转换任务的学习模型。

        参考：

        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023

        [2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros. Image-to-Image Translation with Conditional Adversarial Networks[J]. CVF, 2017. 1125, 1134
        :param input_channel: 输入数据通道，一般是图片通道数。
        :param out_channel: 输出特征通道数，一般是图片通道数。
        :param base_channel: 决定网络复杂度的基础通道数，需为大于0的整数。数值越高决定提取的特征维度越高。
        :param kernel_size: 卷积层使用的感受野大小
        :param bn_momen: 批量标准化层的动量超参数
        """
        cp_layer = lambda i, o: nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(o, momentum=bn_momen)
        )
        ep_layer = lambda i, o: nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.Dropout(dropout, True),
        )
        base_channel = int(base_channel)
        self.contracting_path = [
            nn.Conv2d(input_channel, base_channel, kernel_size=kernel_size, stride=2, padding=1),  # 256^2 -> 128^2
            cp_layer(base_channel, base_channel * 2),  # 128^2 -> 64^2
            cp_layer(base_channel * 2, base_channel * 4),  # 64^2 -> 32^2
            cp_layer(base_channel * 4, base_channel * 8),  # 32^2 -> 16^2
            cp_layer(base_channel * 8, base_channel * 8),  # 16^2 -> 8^2
            cp_layer(base_channel * 8, base_channel * 8),  # 8^2 -> 4^2
            cp_layer(base_channel * 8, base_channel * 8),  # 4^2 -> 2^2
        ]
        self.expanding_path = [
            ep_layer(base_channel * 8, base_channel * 8),  # 2^2 -> 4^2
            ep_layer(base_channel * 16, base_channel * 8),  # 4^2 -> 8^2
            ep_layer(base_channel * 16, base_channel * 8),  # 8^2 -> 16^2
            ep_layer(base_channel * 16, base_channel * 4),  # 16^2 -> 32^2
            ep_layer(base_channel * 8, base_channel * 2),  # 32^2 -> 64^2
            ep_layer(base_channel * 4, base_channel),  # 64^2 -> 128^2
        ]
        self.output_path = [
            ep_layer(base_channel * 2, base_channel),  # 128^2 -> 256^2
            nn.ReLU(True),
            nn.Conv2d(base_channel, out_channel, kernel_size=kernel_size + 1, stride=1, padding=2),
            nn.Tanh()
        ]
        super(UNet256, self).__init__(
            *self.contracting_path, *self.expanding_path, *self.output_path,
            input_size=(input_channel, 256, 256), **kwargs
        )

    def forward(self, input):
        cp_results = []
        for layer in self.contracting_path:
            input = layer(input)
            cp_results.append(input)
        self.bottleneck_opt = cp_results.pop()
        cp_results = reversed(cp_results)  # 需要去除掉最后一个结果
        # cp_results = reversed(cp_results[:-1])  # 需要去除掉最后一个结果
        for layer in self.expanding_path:
            input = layer(input)
            input = torch.hstack((input, next(cp_results)))
        for layer in self.output_path:
            input = layer(input)
        return input
