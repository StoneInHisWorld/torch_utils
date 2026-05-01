import functools

import torch
from torch import nn

from layers.resnet_blocks import ResnetBlock
from networks.basic_nn import BasicNN
from networks.nets.pix2pix import _get_ls_fn, _get_lr_scheduler, _get_optimizer
from networks.nets.unet import UNet128 as UNet128Genarator
from networks.nets.unet import UNet256 as UNet256Genarator


class ResNetGenerator(nn.Sequential):
    """基于Resnet的生成器，在一系列下采样/上采样操作之间插入ResNet块。
    原作者采用torch代码与Justin Johnson的神经网络风格传输项目思想的结合。
    （参见https://github.com/jcjohnson/fast-neural-style）
    """

    def __init__(self, input_channel, output_channel,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """构造一个基于Resnet的生成器

        :param input_channel: 输入图片的通道数
        :param output_channel: 输出图片的通道数
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
            nn.Conv2d(input_channel, ngf, kernel_size=7, padding=0, bias=use_bias),
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
        model += [nn.Conv2d(ngf, output_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.input_size = (input_channel, 256, 256)
        super(ResNetGenerator, self).__init__(*model)


class Pix2Pix_G(BasicNN):

    def __init__(self, version='u256', *layers, **kwargs):
        """适用于图片翻译、转换任务的学习模型。
        参考：

        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023

        [2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros. Image-to-Image Translation with Conditional Adversarial Networks[J]. CVF, 2017. 1125, 1134
        :param version: 指定pix2pix生成器版本的字符串。
            支持['u256', 'r9', 'u128']，要求的图片大小分别为[(256, 256), (256, 256), (128, 128)]
        :param layers: 参见各个生成器的位置参数
            包括UNet256Generator、UNet128Generator、ResNetGenerator
        :param kwargs: 参见各个生成器的关键字参数
        """
        supported = ['u256', 'r9', 'u128']
        device = kwargs.pop("device")
        if version == 'u256':
            model = UNet256Genarator(*layers, **kwargs)
        elif version == 'u128':
            model = UNet128Genarator(*layers, **kwargs)
        elif version == 'r9':
            kwargs['n_blocks'] = 9
            model = ResNetGenerator(*layers, **kwargs)
        else:
            raise NotImplementedError(f'不支持的生成器版本{version}，支持的生成器版本包括{supported}')
        assert "input_size" not in kwargs.keys(), f"{self.__class__.__name__}不支持赋值输入大小！"
        super(Pix2Pix_G, self).__init__(model, device=device, input_size=model.input_size[1:],
                                        **kwargs)

    def _get_ls_fn(self, *ls_args):
        if hasattr(self, "train_ls_fn_s"):
            # 如果本网络已经指定了训练损失函数，则说明此时赋予的是测试损失函数
            return _get_ls_fn(False, self.__class__, *ls_args)
        else:
            return _get_ls_fn(True, self.__class__, *ls_args)

    def _get_optimizer(self, *o_args):
        return _get_optimizer(self, *o_args)

    def _get_lr_scheduler(self, *l_args):
        return _get_lr_scheduler(self.__class__, self.optimizer_s[0], *l_args)

    def _forward_impl(self, X, y):
        X, pred, netD = X
        if torch.is_grad_enabled():
            ls_fn = self.train_ls_fn_s[0]
        else:
            ls_fn = self.test_ls_fn_s[0]
        return None, [*ls_fn(X, y, pred, netD)]

    def _backward_impl(self, *ls_es):
        ls_es[0].backward()