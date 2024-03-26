import functools

import torch
from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.resnet_block import ResnetBlock


class UNet256Genarator(nn.Sequential):

    def __init__(self, input_channel, out_channel,
                 base_channel=4, kernel_size=4, bn_momen=0.8):
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
            nn.Conv2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.ReLU()
        )
        # TODO：会使数据变得越来越大
        ep_layer = lambda i, o: nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(i, o, kernel_size=kernel_size + 1, stride=1, padding=2),
            nn.BatchNorm2d(o, momentum=bn_momen),
            nn.ReLU()
        )
        base_channel = int(base_channel)
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
            nn.Conv2d(base_channel * 2, out_channel, kernel_size=kernel_size + 1, stride=1, padding=2),
        ]
        super(UNet256Genarator, self).__init__(
            *self.expanding_path, *self.contracting_path, *self.output_path
        )

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


class ResNetGenerator(nn.Sequential):
    """基于Resnet的生成器，在一系列下采样/上采样操作之间插入ResNet块。
    原作者采用torch代码与Justin Johnson的神经网络风格传输项目思想的结合。
    （参见https://github.com/jcjohnson/fast-neural-style）
    """

    def __init__(self, input_nc, output_nc,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """构造一个基于Resnet的生成器

        :param input_nc: 输入图片的通道数
        :param output_nc: 输出图片的通道数
        :param ngf: 最后卷积层的过滤层数
        :param use_dropout: 是否使用Dropout()层
        :param n_blocks: ResNet块的数量
        :param padding_type: 卷积区中的padding层类型，可选: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # 加入下采样层
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 增加ResNet块

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # 增加上采样层
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        # self.add_module('model', nn.Sequential(*model))
        # self.model = nn.Sequential(*model)
        super(ResNetGenerator, self).__init__(*model)


class Pix2Pix_G(BasicNN):

    required_shape = (256, 256)

    # def __init__(self, input_channel, out_channel,
    #              base_channel=4, kernel_size=4, bn_momen=0.8,
    #              **kwargs):
    def __init__(self, version='u256', *args, **kwargs):
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
        # cp_layer = lambda i, o: nn.Sequential(
        #     nn.Conv2d(i, o, kernel_size=kernel_size, stride=2, padding=1),
        #     nn.BatchNorm2d(o, momentum=bn_momen),
        #     nn.ReLU()
        # )
        # # TODO：会使数据变得越来越大
        # ep_layer = lambda i, o: nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(i, o, kernel_size=kernel_size + 1, stride=1, padding=2),
        #     nn.BatchNorm2d(o, momentum=bn_momen),
        #     nn.ReLU()
        # )
        # base_channel = int(base_channel)
        # self.contracting_path = [
        #     cp_layer(input_channel, base_channel),
        #     cp_layer(base_channel, base_channel * 2),
        #     cp_layer(base_channel * 2, base_channel * 4),
        #     cp_layer(base_channel * 4, base_channel * 8),
        #     cp_layer(base_channel * 8, base_channel * 8),
        #     cp_layer(base_channel * 8, base_channel * 8),
        #     cp_layer(base_channel * 8, base_channel * 8),
        # ]
        # self.expanding_path = [
        #     ep_layer(base_channel * 8, base_channel * 8),
        #     ep_layer(base_channel * 16, base_channel * 8),
        #     ep_layer(base_channel * 16, base_channel * 8),
        #     ep_layer(base_channel * 16, base_channel * 4),
        #     ep_layer(base_channel * 8, base_channel * 2),
        #     ep_layer(base_channel * 4, base_channel),
        # ]
        # self.output_path = [
        #     ep_layer(base_channel * 2, base_channel * 2),
        #     nn.Conv2d(base_channel * 2, out_channel, kernel_size=kernel_size + 1, stride=1, padding=2),
        # ]
        super(Pix2Pix_G, self).__init__()
        supported = ['u256', 'r9']
        if version == 'u256':
            self.model = UNet256Genarator(*args, **kwargs)
            self.required_shape = (256, 256)
        elif version == 'r9':
            kwargs['n_blocks'] = 9
            self.model = ResNetGenerator(*args, **kwargs)
            self.required_shape = (256, 256)
        else:
            raise NotImplementedError(f'不支持的生成器版本{version}，支持的生成器版本包括{supported}')

    # def forward(self, input):
    #     return self.model(input)
    # def forward(self, input):
    #     cp_results = []
    #     for layer in self.contracting_path:
    #         input = layer(input)
    #         cp_results.append(input)
    #     cp_results = reversed(cp_results[:-1])  # 需要去除掉最后一个结果
    #     for layer in self.expanding_path:
    #         input = layer(input)
    #         input = torch.hstack((input, next(cp_results)))
    #     for layer in self.output_path:
    #         input = layer(input)
    #     return input
