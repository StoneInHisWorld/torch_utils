import functools
from typing import List, Tuple

import torch
from torch import nn

from networks.basic_nn import BasicNN
from layers.identity import Identity
from networks.nets.pix2pix import _get_ls_fn, _get_optimizer, _get_lr_scheduler, _backward_impl


class Pix2Pix_D(BasicNN):
    """适用于图片翻译、转换任务的学习模型pix2pix的分辨器模块"""

    def __init__(self, input_nc, ndf,
                 net_type='basic', n_layers_D=3, norm_type='batch',
                 **kwargs):
        """适用于图片翻译、转换任务的学习模型pix2pix的分辨器模块

        提供三种类型的分辨器:
        [basic]: 原始的pix2pix论文中提到的`PatchGAN`分辨器，能够分辨70x70交叠块的真假。
        相较于全图分辨器，块级分辨器的架构的参数更少，能在全卷积形式中应用于任意形状的图片
        [n_layers]: 该模式可通过`n_layers_D`指定分辨器卷积层数量。
        [pixel]: 1x1 PixelGAN分辨器能分辨像素的真假。对于空间信息无提升效果，但建议使用更高的色域。

        可选三种类型的标准化层：batch | instance | none
        分辨器通过`init_net`进行初始化，并使用`LeakyReLU`进行非线性操作。

        参考：

        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023

        [2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.
        Image-to-Image Translation with Conditional Adversarial Networks[J]. CVF, 2017. 1125, 1134

        :param input_nc: 输入图片的通道数
        :param ndf: 第一卷积层的过滤层数
        :param net_type: 结构名称 -- basic | n_layers | pixel
        :param n_layers_D: 分辨器中的卷积层数，仅当`netD == 'n_layers'`时有效
        :param norm_type: 标准化层的类型
        :return: 返回分辨器
        """
        norm_layer = self.__get_norm_layer(type=norm_type)

        if net_type == 'basic':  # 默认PatchGAN分类器
            net = self.__getNLayerD(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif net_type == 'n_layers':  # 更多选择
            net = self.__getNLayerD(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif net_type == 'pixel':  # 分辨每个像素的真假
            net = self.__getPixelD(input_nc, ndf, norm_layer=norm_layer)
        else:
            raise NotImplementedError(f'不支持的分辨器类型{net_type}!')
        # kwargs['input_size'] = (6, 256, 256) if kwargs.get('input_size') is None else kwargs.get('input_size')
        super(Pix2Pix_D, self).__init__(*net, **kwargs)

    def __get_norm_layer(self, type='instance'):
        """返回一个标准化层

        对于批量标准化层，使用可学习的仿射参数并追踪动态数据（均值/stddev）；对于实例标准化层则不然。

        Parameters:
        :param type: 标准化层的类型: batch | instance | none
        """
        if type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif type == 'none':
            def norm_layer(x):
                return Identity()
        else:
            raise NotImplementedError(f'不支持的标准化层{type}!')
        return norm_layer

    def __getNLayerD(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """构造一个PathGAN分辨器

        :param input_nc: 输入图片的通道数
        :param ndf: 末卷积层的过滤层数
        :param n_layers: 分辨器的卷积层数量
        :param norm_layer: 正则化层类型
        """
        if isinstance(norm_layer, functools.partial):  # 二维批量正则层拥有仿射变量，无需使用偏置
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4  # 感受野窗口大小
        padw = 1  # 装填窗口大小
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐渐增加过滤层
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # 输出单一通道预测映射
        return sequence

    def __getPixelD(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """构造一个1x1 PathGAN分辨器

        :param input_nc: 输入图片的通道数
        :param ndf: 末卷积层的过滤层数
        :param norm_layer: 正则化层类型
        """
        if isinstance(norm_layer, functools.partial):  # 二维批量正则层拥有仿射变量，无需使用偏置
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        return layers

    def _get_ls_fn(self, ls_args: List[Tuple[str, dict]]):
        if hasattr(self, "train_ls_fn_s"):
            # 如果本网络已经指定了训练损失函数，则说明此时赋予的是测试损失函数
            return _get_ls_fn(False, self.__class__, *ls_args)
        else:
            return _get_ls_fn(True, self.__class__, *ls_args)

    def _get_optimizer(self, o_args):
        return _get_optimizer(self, *o_args)

    def _get_lr_scheduler(self, l_args):
        return _get_lr_scheduler(self.__class__, self.optimizer_s[0], *l_args)

    def _forward_impl(self, X, y):
        X, pred = X
        if torch.is_grad_enabled():
            return None, [*self.train_ls_fn_s[0](X, y, pred, self)]
        else:
            return None, []

    def _backward_impl(self, *ls_es):
        ls_es[0].backward()
