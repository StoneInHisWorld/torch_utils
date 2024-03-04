import torch
from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.val2img import Val2Fig


class AdaP2P(BasicNN):

    def __init__(self,
                 input_channels, output_channels,
                 input_shape, compressed_shape, output_shape,
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
        :param input_channels: 输入特征集通道数，一般是图片通道数。
        :param output_channels: 输出通道数，一般是图片通道数。
        :param input_shape: 输入特征集形状。因为受到skip_connection的限制，input_shape值的因数不能含有奇数。
        :param compressed_shape: 压缩特征集的最终形状。
        :param output_shape: 输出特征集形状。网络会根据输出特征集形状拼凑输出路径。
        :param max_channel: 所需的最大通道数。网络会根据输入特征集形状、压缩特征集最终形状拼凑对抗路径（下采样）和扩展路径（上采样）。
        :param base_channel: 决定网络复杂度的基础通道数，需为大于0的整数。数值越高决定提取的特征维度越高。
        :param kernel_size: 卷积层使用的感受野大小。
        :param bn_momen: 批量标准化层的动量超参数。
        :param output_img: 是否要将输出转化为图片。
        :param kwargs: BasicNN关键词参数
        """
        # 下采样对抗路径，每次将长和宽减半
        self.expanding_path = []
        self.contracting_path = []
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
        cur_shape, _, cur_out = self.construct_cp(
            input_shape[0], input_channels, base_channel,
            compressed_shape,
            max_channel,
            cp_layer
        )
        # 拼凑扩展路径
        # TODO： //= 有将shape/out置为0的风险，请解决这个问题，或者弃用本网络，使用pix2pix
        cur_shape, _, cur_out = self.construct_ep(
            compressed_shape, cur_out, cur_out // 2,
            output_channels, len(self.contracting_path),
            ep_layer
        )
        # 拼凑输出路径，确保输出通道数与输入通道数相符，输出形状与指定形状相符
        self.output_path = self.construct_op(
            kernel_size, bn_momen,
            cur_shape // 2, cur_out * 2,
            output_shape, output_channels,
            output_img
        )
        super().__init__(*self.contracting_path, *self.expanding_path, *self.output_path,
                         **kwargs)

    def construct_cp(self,
                     cur_shape, cur_in, cur_out,
                     compress_shape, max_channel,
                     cp_layer):
        self.contracting_path = []
        # 拼凑对抗路径
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
        return cur_shape, cur_in, cur_out

    def construct_ep(self,
                     cur_shape, cur_in, cur_out,
                     output_channel, cp_len,
                     ep_layer
                     ):
        while (cur_out >= output_channel and
               len(self.expanding_path) < cp_len):
            self.expanding_path.append(ep_layer(cur_in, cur_out))
            cur_in = cur_out * 2
            cur_out //= 2
            cur_shape *= 2
        return cur_shape, cur_in, cur_out,

    def construct_op(self,
                     kernel_size, bn_momen,
                     cur_shape, cur_out,
                     output_shape, output_channel,
                     output_img
                     ):
        output_path = []
        # 将形状压缩成输出形状
        if cur_shape < output_shape[0]:
            output_path += [
                nn.Upsample(output_shape),
                # nn.BatchNorm2d(cur_out, momentum=bn_momen),
                # nn.ReLU()
            ]
        while cur_shape > output_shape[0]:
            output_path += [
                nn.Conv2d(cur_out, cur_out, kernel_size=kernel_size, stride=2, padding=1),
                # nn.BatchNorm2d(cur_out, momentum=bn_momen),
                # nn.ReLU()
            ]
            cur_shape //= 2
        # 将通道压缩为所需输出通道
        if cur_out > output_channel:
            output_path += [
                nn.Conv2d(cur_out, output_channel, kernel_size=kernel_size + 1, stride=1, padding=2),
            ]
        output_path += [
            nn.BatchNorm2d(output_channel, momentum=bn_momen),
            nn.ReLU()
        ]
        if output_img is not None:
            output_path.append(Val2Fig(output_img))
        return output_path

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
