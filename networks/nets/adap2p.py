import torch
from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.val2img import Val2Fig


class AdaP2P(BasicNN):

    def __init__(self,
                 input_channel,
                 input_shape, compress_shape, output_shape,
                 max_channel=torch.inf, base_channel=4,
                 kernel_size=4, bn_momen=0.8, output_img=None,
                 **kwargs):
        """
        可自动根据输入特征集形状进行调整的pix2pix网络。可指定输入特征集

        参考：

        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023

        [2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.
            Image-to-Image Translation with Conditional Adversarial Networks[J].
            CVF, 2017. 1125, 1134
        :param input_channel: 输入特征集通道数，一般是图片通道数。
        :param input_shape: 输入特征集形状。
        :param compress_shape: 压缩特征集的最终形状。
        :param output_shape: 输出特征集形状。网络会根据输出特征集形状拼凑输出路径。
        :param max_channel: 所需的最大通道数。
        网络会根据输入特征集形状、压缩特征集最终形状拼凑对抗路径（下采样）和扩展路径（上采样）。
        :param base_channel: 决定网络复杂度的基础通道数，需为大于0的整数。数值越高决定提取的特征维度越高。
        :param kernel_size: 卷积层使用的感受野大小。
        :param bn_momen: 批量标准化层的动量超参数。
        :param output_img: 是否要将输出转化为图片。
        :param kwargs: BasicNN关键词参数
        """
        # 下采样对抗路径，每次将长和宽减半
        cp_layer = lambda i, o: nn.Sequential(
            nn.Conv2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.ReLU()
        )
        # 上采样扩展路径，每次将长和宽拓宽两倍
        ep_layer = lambda i, o: nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(i, o, kernel_size=kernel_size + 1, stride=1, padding=2),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.ReLU()
        )
        # 拼凑对抗路径
        self.contracting_path = []
        cur_shape = input_shape[0]
        cur_in, cur_out = input_channel, base_channel
        while cur_shape >= compress_shape and cur_out <= max_channel:
            self.contracting_path.append(cp_layer(cur_in, cur_out))
            cur_in = cur_out
            cur_out *= 2
            cur_shape //= 2
        cur_out //= 2
        while cur_shape > compress_shape:
            self.contracting_path.append(cp_layer(cur_out, cur_out))
            cur_shape //= 2
        while cur_out <= max_channel != torch.inf:
            self.contracting_path.append(cp_layer(cur_in, cur_out))
            cur_in = cur_out
            cur_out *= 2
        # 拼凑扩展路径
        self.expanding_path = []
        cur_shape = compress_shape
        cur_in, cur_out = cur_out, cur_out // 2
        # while cur_shape <= input_shape[0]:
        # while cur_shape < output_shape[0] and cur_out > input_channel:
        while (cur_out >= input_channel and
               len(self.expanding_path) < len(self.contracting_path)):
            self.expanding_path.append(ep_layer(cur_in, cur_out))
            cur_in = cur_out * 2
            cur_out //= 2
            cur_shape *= 2
        # 拼凑输出路径，确保输出通道数与输入通道数相符，输出形状与指定形状相符
        self.output_path = []
        if cur_shape < output_shape[0]:
            self.output_path += [
                nn.Upsample(output_shape),
            ]
        if cur_out > input_channel:
            self.output_path += [
                nn.Conv2d(cur_out * 2, input_channel, kernel_size=kernel_size + 1, stride=1, padding=2),
            ]
        self.output_path += [
            nn.BatchNorm2d(input_channel, momentum=bn_momen),
            nn.ReLU()
        ]
        if output_img is not None:
            self.output_path.append(Val2Fig(output_img))
        super().__init__(*self.contracting_path, *self.expanding_path, *self.output_path,
                         **kwargs)

    def forward(self, input):
        cp_results = []
        for layer in self.contracting_path:
            input = layer(input)
            cp_results.append(input)
        cp_results = reversed(cp_results[:-1])  # 需要去除掉最后一个结果
        for layer in self.expanding_path:
            input = layer(input)
            try:
                input = torch.hstack((input, next(cp_results)))
            except StopIteration:
                continue
        for layer in self.output_path:
            input = layer(input)
        return input