from typing import List

import torch
import utils.func.torch_tools as ttools

from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.inception import Inception_v1, Inception_v2A, Inception_v2B, Inception_v2C, Inception_v3A, \
    Inception_v3B, Inception_v3C
from networks.layers.multi_output import MultiOutputLayer


class GoogLeNet(BasicNN):

    input_size = (224, 224)

    def __init__(self, in_channels, out_features,
                 version='1', regression=False, dropout_rate=0., bn_momen=0.95,
                 side_headed=True,
                 **kwargs):
        """经典GoogLeNet模型。

        :param in_channels: 输入通道
        :param out_features: 输出特征
        :param device: 设置本网络所处设备
        """
        supported = {'1', '2', '3'}
        vargs = in_channels, dropout_rate  # 版本参数
        self.side_head = None
        if version == '1':
            get_blocks = self.__get_version1
            multi_in = 1024
        elif version == '2':
            get_blocks = self.__get_version2
            multi_in = 2048
        elif version == '3':
            self.side_head = side_headed
            get_blocks = self.__get_version3
            multi_in = 2048
            vargs = in_channels, out_features, regression, dropout_rate, bn_momen
        else:
            raise NotImplementedError(f'暂不支持的GoogLeNet类型{version}，当前支持的类型包括{supported}')
        self.version = version
        super().__init__(
            *get_blocks(*vargs), MultiOutputLayer(
                multi_in, out_features, dropout_rate, bn_momen, regression=regression
            ),
            **kwargs
        )

    @staticmethod
    def __get_version1(in_channels, dropout_rate=0.):
        inception = Inception_v1
        dropout = nn.Dropout(dropout_rate)
        b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            dropout,
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            dropout,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b3 = nn.Sequential(
            inception(192, 64, (96, 128), (16, 32), 32),
            dropout,
            inception(256, 128, (128, 192), (32, 96), 64),
            dropout,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b4 = nn.Sequential(
            inception(480, 192, (96, 208), (16, 48), 64),
            dropout,
            inception(512, 160, (112, 224), (24, 64), 64),
            dropout,
            inception(512, 128, (128, 256), (24, 64), 64),
            dropout,
            inception(512, 112, (144, 288), (32, 64), 64),
            dropout,
            inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b5 = nn.Sequential(
            inception(832, 256, (160, 320), (32, 128), 128),
            dropout,
            inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return b1, b2, b3, b4, b5

    @staticmethod
    def __get_version2(in_channels, dropout_rate=0.):
        """参考：
        [1] https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html

        [2] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J] https://arxiv.org/abs/1512.00567.
        :param in_channels:
        :param dropout_rate:
        :return:
        """
        dropout = nn.Dropout(dropout_rate)
        b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2), nn.ReLU(), dropout,
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(), dropout,
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), dropout,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        b2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3), nn.ReLU(), dropout,
            nn.Conv2d(80, 192, kernel_size=3, stride=2), nn.ReLU(), dropout,
            nn.Conv2d(192, 288, kernel_size=3, padding=1), nn.ReLU(), dropout,
        )
        # 以下参数为自创
        b3 = nn.Sequential(
            Inception_v2A(288, 128, (128, 192), (16, 32, 96), 64), dropout,
            Inception_v2A(480, 192, (96, 208), (16, 32, 48), 64), dropout,
            Inception_v2A(512, 256, (128, 256), (32, 64, 128), 128), dropout,
            nn.MaxPool2d(kernel_size=3, stride=2)  # 形状降维成17x17
        )

        b4 = nn.Sequential(
            Inception_v2B(768, 256, (80, 160, 320), (16, 32, 64, 96, 128), 128), dropout,
            Inception_v2B(832, 256, (80, 160, 320), (16, 32, 64, 96, 128), 128), dropout,
            Inception_v2B(832, 384, (96, 192, 384), (24, 48, 80, 96, 128), 128), dropout,
            Inception_v2B(1024, 384, (96, 192, 384), (24, 48, 80, 96, 128), 128), dropout,
            Inception_v2B(1024, 384, (128, 256, 512), (48, 96, 128, 192, 256), 128), dropout,
            nn.MaxPool2d(kernel_size=3, stride=2)  # 形状降维成7x7
        )

        b5 = nn.Sequential(
            Inception_v2C(1280, 512, (128, 256, 256), (48, 96, 128, 128), 256), dropout,
            Inception_v2C(1536, 768, (160, 320, 320), (48, 96, 128, 128), 384), dropout,  # 总通道数为2048
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return b1, b2, b3, b4, b5

    def __get_version3(self, in_channels, out_features, regression, dropout_rate, bn_momen):
        """参考：
        [1] https://pytorch.org/vision/stable/_modules/torchvision/models/inception.html

        [2] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J] https://arxiv.org/abs/1512.00567.
        :param in_channels:
        :param out_features:
        :param regression:
        :param dropout_rate:
        :param bn_momen:
        :return:
        """
        dropout = nn.Dropout(dropout_rate)
        b1 = nn.Sequential(
            # 输出维度299x299x输入通道数
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2), nn.ReLU(), dropout,
            # 149x149x32
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(), dropout,
            # 147x147x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), dropout,
            # 147x147x64
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 73x73x64
        )
        b2 = nn.Sequential(
            # 73x73x64
            nn.Conv2d(64, 80, kernel_size=1), nn.ReLU(), dropout,
            # 73x73x80
            nn.Conv2d(80, 192, kernel_size=3), nn.ReLU(), dropout,
            # 71x71x192
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 35x35x192
        )
        b3 = nn.Sequential(
            # 35x35x192
            Inception_v3A(192, 64, (48, 64), (64, 96, 96), 32), dropout,
            # 35x35x256
            Inception_v3A(256, 64, (48, 64), (64, 96, 96), 64), dropout,
            # 35x35x288
            Inception_v3A(288, 64, (48, 64), (64, 96, 96), 64), dropout,
            # 35x35x288
        )
        b4 = nn.Sequential(
            # 35x35x288
            Inception_v3B(288, 384, (64, 96, 96)), dropout
            # 17x17x768
        )
        b5 = nn.Sequential(
            # 17x17x768
            Inception_v2B(768, 192, (128, 128, 192), (128, 128, 128, 128, 192), 192), dropout,
            # 17x17x768
            Inception_v2B(768, 192, (160, 160, 192), (160, 160, 160, 160, 192), 192), dropout,
            # 17x17x768
            Inception_v2B(768, 192, (160, 160, 192), (160, 160, 160, 160, 192), 192), dropout,
            # 17x17x768
            Inception_v2B(768, 192, (192, 192, 192), (192, 192, 192, 192, 192), 192), dropout,
            # 17x17x768
        )
        if self.side_head:
            # version3使用的辅助Softmax分类器
            side_head = nn.Sequential(
                # 17x17x768
                nn.AvgPool2d(5, stride=3),
                # 5x5x768
                nn.Conv2d(768, 128, kernel_size=1),
                # 5x5x128
                nn.Conv2d(128, 768, kernel_size=5),
                # 1x1x768
                nn.AdaptiveAvgPool2d(1),
                # 1x1x768
                MultiOutputLayer(768, out_features, dropout_rate,
                                 momentum=bn_momen, regression=regression)
                # out_features x 1
            )
            # 为head设置参数
            side_head[2].stddev = 0.01
            side_head[4].stddev = 0.001
        b6 = nn.Sequential(
            # 17x17x768
            Inception_v3C(768, (192, 320), (192, 192, 192, 192)), dropout
            # 8x8x1280
        )
        b7 = nn.Sequential(
            # 8x8x1280
            Inception_v2C(1280, 320, (384, 384, 384), (448, 384, 384, 384), 192), dropout,
            # 8x8x2048
            Inception_v2C(2048, 320, (384, 384, 384), (448, 384, 384, 384), 192), dropout,
            # 8x8x2048
        )
        b8 = nn.Sequential(
            # 8x8x2048
            nn.AvgPool2d(8), nn.Dropout(0.2), nn.Flatten()
            # 1x2048
        )
        if self.side_head:
            return b1, b2, b3, b4, b5, b6, b7, b8, side_head
        else:
            return b1, b2, b3, b4, b5, b6, b7, b8

    @staticmethod
    def get_required_shape(version='1'):
        supported = ['1', '2', '3']
        if version == '1':
            return 224, 224
        elif version == '2' or version == '3':
            return 299, 299
        else:
            raise NotImplementedError(f'不支持的版本{version}，目前支持{supported}')

    def forward_backward(self, X, y, backward=True):
        """前向及反向传播
        通过改变训练状态来指示模型是否使用辅助分类器，以及进行损失值的计算。
        :param X: 特征集
        :param y: 标签集
        :param backward: 是否进行反向传播
        :return: 预测值，损失值集合
        """
        pre_state = self.is_train
        self.is_train = backward and self.is_train
        ret = super().forward_backward(X, y, backward)
        self.is_train = pre_state
        return ret

    def _forward_impl(self, X, y):
        """前向传播实现。
        针对版本3实现的前向传播，主要针对辅助分类器。
        如果开启了辅助分类器，即self.side_head==True，则会将本分类器和辅助分类器的预测损失值相加，同时反向传播。
        :param X: 特征集批。
        :param y: 标签集批。
        :return: 总预测值，[总预测损失值+辅助预测损失值，位于各种要求损失函数下]
        """
        if self.side_head is not None:
            # 如果选择了使用辅助分类器
            for i, m in enumerate(self):
                if i == 4:
                    X = m(X)
                    side_head_input = X
                elif i == 8:
                    pass
                else:
                    X = m(X)
            if self.is_train:
                aux = self[8](side_head_input)
                gross_ls_fn, aux_ls_fn, ls_fn = self._ls_fn_s
                ls_es = [gross_ls_fn(X, aux, y), aux_ls_fn(aux, y), ls_fn(X, y)]
            else:
                ls_es = [self._ls_fn_s[2](X, y)]
            return X, ls_es
        else:
            return super()._forward_impl(X, y)

    def _init_submodules(self, init_str, **kwargs):
        """若指定初始化类型为`original`，则会使用论文中的方法初始化网络各模块。
        :param init_str: 初始化方法
        :param kwargs: 初始化方法关键词参数
        :return: None
        """
        if init_str == 'original':
            # 如果不使用预先保存的参数
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            super()._init_submodules(init_str, **kwargs)

    def _get_ls_fn(self, *args):
        """若指定损失函数类型为`original`，则使用论文中的标签平滑损失计算方式，辅助分类器的损失值权重为0.4
        :param args: 多个二元组，元组格式为（损失函数类型字符串，本损失函数关键词参数）
        :return: 损失函数序列
        """
        ls_fn_s, loss_names, test_ls_names = [], [], []
        for (type_s, kwargs) in args:
            size_averaged = kwargs.pop('size_averaged', True)
            if type_s == 'original' and self.side_head is not None:
                ls_fn = ttools.get_ls_fn('entro', label_smoothing=0.1)
                unwrapped_fns = [
                    lambda pred, aux, y: ls_fn(aux, y) * 0.4 + ls_fn(pred, y) * 0.6,
                    lambda aux, y: ls_fn(aux, y),
                    lambda pred, y: ls_fn(pred, y)
                ]
                ls_fn_s += unwrapped_fns if size_averaged else [
                    lambda x, y: ttools.sample_wise_ls_fn(x, y, fn)
                    for fn in unwrapped_fns
                ]
                loss_names += ['GROSS_ENTRO', 'AUX_ENTRO', 'MAIN_ENTRO']
                test_ls_names += ['ENTRO']
            else:
                ls_fn, ls_names, tls_names = super()._get_ls_fn((type_s, kwargs))
                ls_fn_s += ls_fn
                loss_names += ls_names
                test_ls_names += tls_names
        return ls_fn_s, loss_names, test_ls_names

    def _get_optimizer(self, *args) -> torch.optim.Optimizer or List[torch.optim.Optimizer]:
        """若指定类型参数为`original`，则使用论文中的优化器RMSPROP，指定的关键字参数会被覆盖
        :param args: 多个二元组，元组格式为（优化器类型字符串，本优化器关键词参数）
        :return: 优化器序列
        """
        lr_names, optimizers = [], []
        for i, (type_s, kwargs) in enumerate(args):
            if type_s == 'original' and self.version == '3':
                type_s = 'rmsprop'
                kwargs = {'lr': 0.045, 'w_decay': 0.9, 'alpha': 0.1}
            optims, lns = super()._get_optimizer((type_s, kwargs), )
            optimizers += optims
            lr_names += lns
            # optimizers.append(ttools.get_optimizer(self, type_s, **kwargs))
            # lr_names.append(f'LR_{i}')
        return optimizers, lr_names

    def _gradient_clipping(self):
        if self.version == '2' or self.version == '3':
            nn.utils.clip_grad_norm_(self.parameters(), 2.)

    def _get_lr_scheduler(self, *args):
        """若指定学习率规划器类型为`original`，则会使用论文中的学习率规划器。
        :param args: 多个二元组，元组格式为（规划器类型字符串，本规划器关键词参数）
        :return: 规划器序列
        """
        schedulers = []
        # if len(args) == 0 and (self.version == '2' or self.version == '3'):
        #     for optimizer in self._optimizer_s:
        #         basic_lr = optimizer.param_groups[0]['lr']
        #         schedulers.append(ttools.get_lr_scheduler(
        #             optimizer, 'lambda',
        #             lr_lambda=lambda epoch: basic_lr ** (0.94 * (epoch // 2))
        #         ))
        for (type_s, kwargs), optimizer in zip(args, self._optimizer_s):
            if type_s == 'original' and (self.version == '2' or self.version == '3'):
                type_s = 'lambda'
                basic_lr = optimizer.param_groups[0]['lr']
                kwargs['lr_lambda'] = lambda epoch: basic_lr ** (0.94 * (epoch // 2))
            schedulers.append(ttools.get_lr_scheduler(
                optimizer, type_s, **kwargs
            ))
        return schedulers

