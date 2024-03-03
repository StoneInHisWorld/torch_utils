import functools

from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.identity import Identity


class Pix2Pix_D(BasicNN):
    required_shape = (256, 256)

    def __init__(self, input_nc, ndf, net_type='basic',
                 n_layers_D=3, norm_type='batch',
                 **kwargs):
        """适用于图片翻译、转换任务的学习模型pix2pix的分辨器模块。

        参考：

        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023

        [2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.
        Image-to-Image Translation with Conditional Adversarial Networks[J]. CVF, 2017. 1125, 1134

        提供三种类型的分辨器:

        [basic]: 原始的pix2pix论文中提到的`PatchGAN`分辨器，能够分辨70x70交叠块的真假。
        相较于全图分辨器，块级分辨器的架构的参数更少，能在全卷积形式中应用于任意形状的图片

        [n_layers]: 该模式可通过`n_layers_D`指定分辨器卷积层数量。

        [pixel]: 1x1 PixelGAN分辨器能分辨像素的真假。对于空间信息无提升效果，但建议使用更高的色域。

        分辨器通过`init_net`进行初始化，并使用`LeakyReLU`进行非线性操作。
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
        if type(norm_layer) == functools.partial:  # 二维批量正则层拥有仿射变量，无需使用偏置
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
        nf_mult_prev = 1
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
        if type(norm_layer) == functools.partial:  # 二维批量正则层拥有仿射变量，无需使用偏置
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


