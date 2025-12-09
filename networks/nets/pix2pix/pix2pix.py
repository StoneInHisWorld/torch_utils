import warnings
from collections import OrderedDict
from typing import Iterable

import torch

from networks.basic_nn import BasicNN
from networks.nets.pix2pix.pix2pix_d import Pix2Pix_D
from networks.nets.pix2pix.pix2pix_g import Pix2Pix_G


class Pix2Pix(BasicNN):
    """通过给定数据对学习输入图片到输出图片的映射，适用于图片翻译、转换任务的学习模型。

    参考论文：
    [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023
    [2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.
       Image-to-Image Translation with Conditional Adversarial Networks[J].
       CVF, 2017. 1125, 1134
   """

    def __init__(self,
                 g_args, g_kwargs,
                 d_args, d_kwargs,
                 direction='AtoB', isTrain=True,
                 **kwargs):
        r"""实现pix2pix模型，通过给定数据对学习输入图片到输出图片的映射。
        pix2pix不使用图片缓存。
        生成器目标函数为 :math:`\ell_G = G_GAN + \lambda_{L1}||G(A)-B||_1`
        分辨器目标函数为 :math:`\ell_D = 0.5 D_read + 0.5 D_fake`
        目前可指定的损失函数包括：pcc | cGAN
        训练过程中输出四个目标值，分别为G_GAN、G_L1、D_real、D_fake：

        - :math:`G_GAN`：生成器的GAN损失
        - :math:`G_{L1}`：生成器的L1损失
        - :math:`D_real`：分辨器分辨真图为真的概率，其输入为真实标签。
        - :math:`D_fake`：分辨器分辨合成图为假的概率，其输入为生成器合成图。

        :param g_args: 生成器位置参数，包括version, *args, **kwargs
            version: 指定pix2pix生成器版本的字符串，支持['u256', 'r9', 'u128']
                三种版本要求的图片大小分别为[(256, 256), (256, 256), (128, 128)]；
            args: 参见各个生成器的位置参数，包括UNet256Generator、UNet128Generator、ResNetGenerator
        :param g_kwargs: 生成器关键词参数，参见各个生成器的关键字参数以及BasicNN关键字参数
        :param d_args: 分辨器位置参数
            input_nc: 输入图片的通道数；
            ndf: 第一卷积层的过滤层数；
            net_type: 结构名称 -- basic | n_layers | pixel；
            n_layers_D: 分辨器中的卷积层数，仅当`netD == 'n_layers'`时有效；
            norm_type: 标准化层的类型，包括 batch | instance | none
        :param d_kwargs: 分辨器关键词参数，请查询BasicNN关键字参数
        :param direction: 方向，'AtoB'意为从特征集预测到标签集，'BtoA'意为从标签集预测到特征集
        :param kwargs: BasicNN关键词参数
        """
        self.direction = direction
        device = torch.device('cpu') if 'device' not in kwargs.keys() else kwargs['device']
        g_kwargs.update({"device": device if "device" not in g_kwargs.keys() else g_kwargs["device"]})
        d_kwargs.update({"device": device if "device" not in d_kwargs.keys() else d_kwargs["device"]})
        netG = Pix2Pix_G(*g_args, **g_kwargs)

        kwargs['input_size'] = netG.input_size[1:]
        if isTrain:
            # 定义一个分辨器
            # conditional GANs需要输入和输出图片，因此分辨器的通道数为input_nc + output_nc
            assert "input_size" not in d_kwargs.keys(), f"{Pix2Pix_D.__name__}不支持赋值输入大小！"
            d_kwargs['input_size'] = (g_args[1] + g_args[2], *netG.input_size[2:])
            netD = Pix2Pix_D(g_args[1] + g_args[2], *d_args, **d_kwargs)
            super(Pix2Pix, self).__init__(OrderedDict([
                ('netG', netG), ('netD', netD)
            ]), **kwargs)
        else:
            super(Pix2Pix, self).__init__(netG, **kwargs)

    def activate(self, is_train: bool,
                 o_args: Iterable, l_args: Iterable, tr_ls_args: Iterable,
                 ts_ls_args: Iterable):
        super().activate(is_train, o_args, l_args, tr_ls_args, ts_ls_args)
        if is_train:
            if len(self.optimizer_s) > 0:
                warnings.warn(f"{self.__class__.__name__}不接受优化器参数！"
                              f"将去除已赋值的优化器，并使用生成器和分辨器的学习率名称")
                self.optimizer_s = []
            self.lr_names = self.netG.lr_names + self.netD.lr_names
            if len(self.scheduler_s) > 0:
                warnings.warn(f"{self.__class__.__name__}不接受学习率规划器参数！"
                              f"将去除已赋值的学习率规划器")
                self.scheduler_s = []
            if len(self.train_ls_fn_s) > 0:
                warnings.warn(f"{self.__class__.__name__}不接受训练损失函数参数！"
                              f"将去除已赋值的训练损失函数，并使用生成器和分辨器的训练损失函数名称")
                self.train_ls_fn_s = []
            self.train_ls_names = self.netG.train_ls_names + self.netD.train_ls_names
        if len(self.test_ls_fn_s) > 0:
            warnings.warn(f"{self.__class__.__name__}不接受测试损失函数参数！"
                          f"将去除已赋值的测试损失函数，并使用生成器和分辨器的测试损失函数名称")
            self.test_ls_fn_s = []
        self.test_ls_names = self.netG.test_ls_names

    def get_lr_groups(self):
        return self.lr_names, [
            [param['lr'] for param in optimizer.param_groups]
            for optimizer in self.netG.optimizer_s + self.netD.optimizer_s
        ]

    def update_lr(self):
        for scheduler in self.netG.scheduler_s + self.netD.scheduler_s:
            scheduler.step()

    def forward(self, input):
        """前向传播
        计算生成器的预测值。
        :param input: 输入特征批
        :return: 生成器预测图片批
        """
        return self.netG(input)

    # def forward_backward(self, X, y):
    #     # 前向传播
    #     assert X.shape == y.shape, (f"Pix2Pix要求输入的标签集数据与特征集数据形状相同，"
    #                                 f"然而得到的输入特征集形状为{X.shape}，标签集形状为{y.shape}")
    #     AtoB = self.direction == 'AtoB'
    #     X, y = [X, y] if AtoB else [y, X]
    #     pred = self(X)
    #     if backward:
    #         self.netD.requires_grad_(True)
    #         _, D_ls = self.netD.forward_backward((X, pred), y)
    #         self.netD.requires_grad_(False)
    #         _, G_ls = self.netG.forward_backward((X, pred, self.netD), y)
    #         ls_es = (*G_ls, *D_ls)
    #     else:
    #         with torch.no_grad():
    #             _, G_ls = self.netG.forward_backward((X, pred, self.netD), y)
    #             ls_es = (*G_ls, )
    #     return pred, ls_es

    def _train(self, X, y):
        # 前向传播
        assert X.shape == y.shape, (f"Pix2Pix要求输入的标签集数据与特征集数据形状相同，"
                                    f"然而得到的输入特征集形状为{X.shape}，标签集形状为{y.shape}")
        AtoB = self.direction == 'AtoB'
        X, y = [X, y] if AtoB else [y, X]
        pred = self(X)
        self.netD.requires_grad_(True)
        _, D_ls = self.netD.forward_backward((X, pred), y)
        self.netD.requires_grad_(False)
        _, G_ls = self.netG.forward_backward((X, pred, self.netD), y)
        return pred, [*G_ls, *D_ls]

    def _predict(self, X, y):
        # 前向传播
        assert X.shape == y.shape, (f"Pix2Pix要求输入的标签集数据与特征集数据形状相同，"
                                    f"然而得到的输入特征集形状为{X.shape}，标签集形状为{y.shape}")
        AtoB = self.direction == 'AtoB'
        X, y = [X, y] if AtoB else [y, X]
        pred = self(X)
        _, G_ls = self.netG.forward_backward((X, pred, self.netD), y)
        return pred, (*G_ls,)

