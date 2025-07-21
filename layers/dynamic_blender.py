import math

from torch import nn as nn


class DynamicBlender(nn.Module):
    """动态晕染器
    存有可学习参数，用于将稀疏采样的像素向量扩充至所需维度，并以二维形式组织
    """

    def __init__(self, input_size: int,
                 tgt_size: int = 224, start_size=7, start_channels=64,
                 output_channel=3):
        """动态晕染器。存有可学习参数，用于将稀疏采样的像素向量扩充至所需维度，并以二维形式组织

        :param input_size: 指定输入向量维度
        :param tgt_size: 指定目标输出形状
        :param start_size: 指定扩充开始维度。
            晕染器会先将向量扩充至start_size * start_size * start_channels维，
            然后在前向传播途中压缩成（批量大小，start_channels，start_size，start_size）
        :param start_channels: 指定扩充开始通道数。晕染器持有的反置卷积层会逐步压缩通道数，直到张量通道数为output_channel。
        :param output_channel: 指定输出张量的通道数
        :return: 晕染器可以作为可调用对象，返回形状为（批量大小，output_channels，tgt_size，tgt_size）的张量
        """
        super().__init__()
        # 计算线性层输出维度
        linear_dim = start_channels * start_size * start_size

        self.input_size = (-1, input_size)
        self.linear = nn.Linear(input_size, linear_dim)
        self.initial_shape = (start_channels, start_size, start_size)

        # 计算需要上采样的次数
        scale_factor = tgt_size // start_size
        num_upsamples = int(math.log2(scale_factor))

        # 动态创建转置卷积层
        layers = []
        in_ch = start_channels
        for i in range(num_upsamples):
            out_ch = max(in_ch // 2, output_channel)  # 通道数逐层减半，但不少于输出通道
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ]
            in_ch = out_ch

        # 确保最终通道数为output_channel
        if in_ch != output_channel:
            layers.append(nn.Conv2d(in_ch, output_channel, kernel_size=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, *self.initial_shape)
        return self.decoder(x)
