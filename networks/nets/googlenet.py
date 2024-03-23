from abc import abstractmethod
from collections.abc import Iterable
from typing import Sized, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

from networks.layers.multi_output import MultiOutputLayer, linear_output
from networks.basic_nn import BasicNN


# class Inception(nn.Module):
#     # c1--c4是每条路径的输出通道数
#     def __init__(self, in_channels, c1, c2, c3, c4):
#         """构成GoogLeNet的Inception块，版本1
#
#         参考：
#
#         [1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
#         Vincent Vanhoucke, Andrew Rabinovich. Going Deeper with Convolutions[J]. https://arxiv.org/abs/1409.4842，2014
#         :param in_channels: 输入通道
#         :param c1: path1中层的感受野，len(c1) == 1
#         :param c2: path2中层的感受野，len(c2) == 2
#         :param c3: path3中层的感受野，len(c3) == 2
#         :param c4: path4中层的感受野，len(c4) == 1
#         """
#         assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
#         assert len(c2) == 2, f'第二条路径的输出通道数c2只能指定2个，然而收到了{len(c2)}个'
#         assert len(c3) == 2, f'第三条路径的输出通道数c3只能指定2个，然而收到了{len(c3)}个'
#         assert not isinstance(c4, Sized), f'第四条路径的输出通道数c4只能为数字！'
#         super(Inception, self).__init__()
#         # 线路1，单1x1卷积层
#         self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
#         # 线路2，1x1卷积层后接3x3卷积层
#         self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
#         self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
#         # 线路3，1x1卷积层后接5x5卷积层
#         self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
#         self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
#         # 线路4，3x3最大汇聚层后接1x1卷积层
#         self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#         self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
#
#     def forward(self, x):
#         p1 = F.relu(self.p1_1(x))
#         p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
#         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
#         p4 = F.relu(self.p4_2(self.p4_1(x)))
#         # 在通道维度上连结输出
#         return torch.cat((p1, p2, p3, p4), dim=1)

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, *args):
        """构成GoogLeNet的Inception块，版本1

        参考：

        [1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
        Vincent Vanhoucke, Andrew Rabinovich. Going Deeper with Convolutions[J]. https://arxiv.org/abs/1409.4842，2014
        :param in_channels: 输入通道
        :param c1: path1中层的感受野，len(c1) == 1
        :param c2: path2中层的感受野，len(c2) == 2
        :param c3: path3中层的感受野，len(c3) == 2
        :param c4: path4中层的感受野，len(c4) == 1
        """
        self.check_para(*args)
        super(Inception, self).__init__()
        self.paths = self.get_paths(in_channels, *args)

    @abstractmethod
    def check_para(self, *args):
        pass

    @abstractmethod
    def get_paths(self, in_channels, *args) -> Tuple[List[nn.Module]]:
        pass

    def forward(self, x):
        results_of_paths = []
        for p in self.paths:
            result = x
            for layer in p:
                result = F.relu(layer(result))
            results_of_paths.append(result)
        return torch.cat(results_of_paths, dim=1)


class Inception_v1(Inception):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        """构成GoogLeNet的Inception块，版本1

        参考：

        [1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
        Vincent Vanhoucke, Andrew Rabinovich. Going Deeper with Convolutions[J].
        https://arxiv.org/abs/1409.4842，2014
        :param in_channels: 输入通道
        :param c1: path1中层的感受野，len(c1) == 1
        :param c2: path2中层的感受野，len(c2) == 2
        :param c3: path3中层的感受野，len(c3) == 2
        :param c4: path4中层的感受野，len(c4) == 1
        """
        super(Inception_v1, self).__init__(in_channels, c1, c2, c3, c4)

    def check_para(self, *args):
        assert len(args) == 4, f'需要为{self.__class__.__name__}提供四个通道参数，但是收到了{len(args)}'
        c1, c2, c3, c4 = args
        assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
        assert len(c2) == 2, f'第二条路径的输出通道数c2只能指定2个，然而收到了{len(c2)}个'
        assert len(c3) == 2, f'第三条路径的输出通道数c3只能指定2个，然而收到了{len(c3)}个'
        assert not isinstance(c4, Sized), f'第四条路径的输出通道数c4只能为数字！'

    def get_paths(self, in_channels, *args):
        c1, c2, c3, c4 = args
        # 线路1，单1x1卷积层
        p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
        self.p1 = p1[0]
        # 线路2，1x1卷积层后接3x3卷积层
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        ]
        self.p2_1, self.p2_2 = p2
        # 线路3，1x1卷积层后接5x5卷积层
        p3 = [
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        ]
        self.p3_1, self.p3_2 = p3
        # 线路4，3x3最大汇聚层后接1x1卷积层
        p4 = [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1)
        ]
        self.p4_1, self.p4_2 = p4
        return p1, p2, p3, p4


class Inception_v2A(Inception):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        """构成GoogLeNet的Inception块，版本2。结构来自于论文中的图5

        参考：

        [1]https://zhuanlan.zhihu.com/p/194382937. 2020.08.22/2024.03.20

        [2] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中层的感受野，len(c1) == 1
        :param c2: path2中层的感受野，len(c2) == 2
        :param c3: path3中层的感受野，len(c3) == 3
        :param c4: path4中层的感受野，len(c4) == 1
        """
        super(Inception_v2A, self).__init__(in_channels, c1, c2, c3, c4)
        # # 线路1，单1x1卷积层，不改变形状
        # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # # 线路2，1x1卷积层后接3x3卷积层，不改变形状
        # self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        # self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        # self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        # self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)
        # self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
        # # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        # self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def check_para(self, *args):
        assert len(args) == 4, f'需要为{self.__class__.__name__}提供四个通道参数，但是收到了{len(args)}'
        c1, c2, c3, c4 = args
        assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
        assert len(c2) == 2, f'第一条路径的输出通道数c2只能指定2个，然而收到了{len(c2)}个！'
        assert len(c3) == 3, f'第一条路径的输出通道数c3只能指定3个，然而收到了{len(c3)}个！'
        assert not isinstance(c4, Sized), f'第四条路径的输出通道数c4只能为数字！'

    def get_paths(self, in_channels, *args) -> Tuple[List[nn.Module]]:
        c1, c2, c3, c4 = args
        # 线路1，单1x1卷积层，不改变形状
        p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
        self.p1 = p1[0]
        # # 线路2，1x1卷积层后接3x3卷积层，不改变形状
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        ]
        self.p2_1, self.p2_2 = p2
        # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        p3 = [
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
            nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
        ]
        self.p3_1, self.p3_2, self.p3_3 = p3
        # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        p4 = [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1)
        ]
        self.p4_1, self.p4_2 = p4
        return p1, p2, p3, p4

    # def __original_layers(self, in_channels, c1, c2, c3, c4):
    #     """对应论文中的图5
    #     :param in_channels:
    #     :param c1:
    #     :param c2:
    #     :param c3:
    #     :param c4:
    #     :return:
    #     """
    #     # # 线路1，单1x1卷积层，不改变形状
    #     # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
    #     # # 线路2，1x1卷积层后接3x3卷积层，不改变形状
    #     # self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
    #     # self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
    #     # # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
    #     # self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
    #     # self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1)
    #     # self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
    #     # # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
    #     # self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    #     # self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    #     # 线路1，单1x1卷积层，不改变形状
    #     p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
    #     # # 线路2，1x1卷积层后接3x3卷积层，不改变形状
    #     p2 = [
    #         nn.Conv2d(in_channels, c2[0], kernel_size=1),
    #         nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
    #     ]
    #     # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
    #     p3 = [
    #         nn.Conv2d(in_channels, c3[0], kernel_size=1),
    #         nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
    #         nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
    #     ]
    #     # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
    #     p4 = [
    #         nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    #         nn.Conv2d(in_channels, c4, kernel_size=1)
    #     ]
    #     return p1, p2, p3, p4

    # def __efficient_layers(self, in_channels, c1, c2, c3, c4, n=7):
    #     """对应论文中的图6
    #     :param in_channels:
    #     :param c1:
    #     :param c2:
    #     :param c3:
    #     :param c4:
    #     :param n:
    #     :return:
    #     """
    #     # # 线路1，单1x1卷积层，不改变形状
    #     # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
    #     # # 线路2，1x1卷积层后接3x3卷积层，不改变形状
    #     # self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
    #     # self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(1, n), padding=1)
    #     # self.p2_3 = nn.Conv2d(c2[1], c2[2], kernel_size=(n, 1), padding=1)
    #     # # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
    #     # self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
    #     # self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(1, n), padding=1)
    #     # self.p3_3 = nn.Conv2d(c3[1], c3[2], kernel_size=(n, 1), padding=1)
    #     # self.p3_4 = nn.Conv2d(c3[2], c3[3], kernel_size=(1, n), padding=1)
    #     # self.p3_5 = nn.Conv2d(c3[3], c3[4], kernel_size=(n, 1), padding=1)
    #     # # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
    #     # self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    #     # self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    #     # 线路1，单1x1卷积层，不改变形状
    #     p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
    #     # 线路2，1x1卷积层后接3x3卷积层，不改变形状
    #     p2 = [
    #         nn.Conv2d(in_channels, c2[0], kernel_size=1),
    #         nn.Conv2d(c2[0], c2[1], kernel_size=(1, n), padding=(1, n // 2)),
    #         nn.Conv2d(c2[1], c2[2], kernel_size=(n, 1), padding=(n // 2, 1))
    #     ]
    #     # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
    #     p3 = [
    #         nn.Conv2d(in_channels, c3[0], kernel_size=1),
    #         nn.Conv2d(c3[0], c3[1], kernel_size=(1, n), padding=(1, n // 2)),
    #         nn.Conv2d(c3[1], c3[2], kernel_size=(n, 1), padding=(n // 2, 1)),
    #         nn.Conv2d(c3[2], c3[3], kernel_size=(1, n), padding=(1, n // 2)),
    #         nn.Conv2d(c3[3], c3[4], kernel_size=(n, 1), padding=(n // 2, 1))
    #     ]
    #     # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
    #     p4 = [
    #         nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    #         nn.Conv2d(in_channels, c4, kernel_size=1)
    #     ]
    #     return p1, p2, p3, p4

    # def __reduced_layers(self, in_channels, c1, c2, c3, c4):
    #     """对应论文中的图7
    #     :param in_channels:
    #     :param c1:
    #     :param c2:
    #     :param c3:
    #     :param c4:
    #     :param n:
    #     :return:
    #     """
    #     # 线路1，单1x1卷积层，不改变形状
    #     self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
    #     # 线路2，1x1卷积层后接3x3卷积层，不改变形状
    #     self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
    #     self.p2_21 = nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1))
    #     self.p2_22 = nn.Conv2d(c2[0], c2[2], kernel_size=(3, 1), padding=(1, 0))
    #     # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
    #     self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
    #     self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(3, 3), padding=1)
    #     self.p3_31 = nn.Conv2d(c3[1], c3[2], kernel_size=(1, 3), padding=(0, 1))
    #     self.p3_32 = nn.Conv2d(c3[1], c3[3], kernel_size=(3, 1), padding=(1, 0))
    #     # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
    #     self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    #     self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    #
    #     def forward(x):
    #         p1 = F.relu(self.p1_1(x))
    #         p2 = F.relu(self.p2_1(x))
    #         p21 = F.relu(self.p2_21(p2))
    #         p22 = F.relu(self.p2_22(p2))
    #         p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
    #         p31 = F.relu(self.p3_31(p3))
    #         p32 = F.relu(self.p3_32(p3))
    #         p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
    #         return torch.cat((p1, p21, p22, p31, p32, p4), dim=1)
    #
    #     self.forward = forward
        # # 线路1，单1x1卷积层，不改变形状
        # p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
        # # 线路2，1x1卷积层后接3x3卷积层，不改变形状
        # p2 = [
        #     nn.Conv2d(in_channels, c2[0], kernel_size=1),
        #     nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1)),
        #     nn.Conv2d(c2[0], c2[2], kernel_size=(3, 1), padding=(1, 0))
        # ]
        # # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        # p3 = [
        #     nn.Conv2d(in_channels, c3[0], kernel_size=1),
        #     nn.Conv2d(c3[0], c3[1], kernel_size=(3, 3), padding=1),
        #     nn.Conv2d(c3[1], c3[2], kernel_size=(1, 3), padding=(0, 1)),
        #     nn.Conv2d(c3[1], c3[3], kernel_size=(3, 1), padding=(1, 0))
        # ]
        # # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        # p4 = [
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels, c4, kernel_size=1)
        # ]
        # return p1, p2, p3, p4


class Inception_v2B(Inception):

    def __init__(self, in_channels, c1, c2, c3, c4, n=7):
        """构成GoogLeNet的Inception块，版本2。结构来自于论文中的图6

        参考：

        [1]https://zhuanlan.zhihu.com/p/194382937. 2020.08.22/2024.03.20

        [2] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中层的感受野，len(c1) == 1
        :param c2: path2中层的感受野，len(c2) == 3
        :param c3: path3中层的感受野，len(c3) == 5
        :param c4: path4中层的感受野，len(c4) == 1
        :param n: path2、3中分解的层的感受野，n最好为大于0的奇数
        """
        super(Inception_v2B, self).__init__(in_channels, c1, c2, c3, c4, n)

    def check_para(self, *args):
        assert len(args) == 5, f'需要为{self.__class__.__name__}提供五个通道参数，但是收到了{len(args)}'
        c1, c2, c3, c4, n = args
        assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
        assert len(c2) == 3, f'第二条路径的输出通道数c2只能指定3个，然而收到了{len(c2)}个！'
        assert len(c3) == 5, f'第三条路径的输出通道数c3只能指定5个，然而收到了{len(c3)}个！'
        assert not isinstance(c4, Sized), f'第四条路径的输出通道数c4只能为数字！'
        assert n > 0, f'n值应该大于0，但是收到了{n}！'

    def get_paths(self, in_channels, *args) -> Tuple[List[nn.Module]]:
        c1, c2, c3, c4, n = args
        # 线路1，单1x1卷积层，不改变形状
        p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
        self.p1 = p1[0]
        # 线路2，1x1卷积层后接3x3卷积层，不改变形状
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c2[1], c2[2], kernel_size=(n, 1), padding=(n // 2, 0))
        ]
        self.p2_1, self.p2_2, self.p2_3 = p2
        # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        p3 = [
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.Conv2d(c3[0], c3[1], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c3[1], c3[2], kernel_size=(n, 1), padding=(n // 2, 0)),
            nn.Conv2d(c3[2], c3[3], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c3[3], c3[4], kernel_size=(n, 1), padding=(n // 2, 0))
        ]
        self.p3_1, self.p3_2, self.p3_3, self.p3_4, self.p3_5 = p3
        # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        p4 = [
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1)
        ]
        self.p4_1, self.p4_2 = p4
        return p1, p2, p3, p4


class Inception_v2C(Inception):

    def __init__(self, in_channels, c1, c2, c3, c4):
        """构成GoogLeNet的Inception块，版本2，结构来自于论文中的图7

        参考：

        [1]https://zhuanlan.zhihu.com/p/194382937. 2020.08.22/2024.03.20

        [2] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中层的感受野，len(c1) == 1
        :param c2: path2中层的感受野，len(c2) == 3
        :param c3: path3中层的感受野，len(c3) == 4
        :param c4: path4中层的感受野，len(c4) == 1
        """
        super(Inception_v2C, self).__init__(in_channels, c1, c2, c3, c4)

    def check_para(self, *args):
        assert len(args) == 4, f'需要为{self.__class__.__name__}提供4个通道参数，但是收到了{len(args)}'
        c1, c2, c3, c4 = args
        assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
        assert len(c2) == 3, f'第二条路径的输出通道数c2只能指定3个，然而收到了{len(c2)}个！'
        assert len(c3) == 4, f'第三条路径的输出通道数c3只能指定4个，然而收到了{len(c3)}个！'
        assert not isinstance(c4, Sized), f'第四条路径的输出通道数c4只能为数字！'

    def get_paths(self, in_channels, *args):
        c1, c2, c3, c4 = args
        # 线路1，单1x1卷积层，不改变形状
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层，不改变形状
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_21 = nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1))
        self.p2_22 = nn.Conv2d(c2[0], c2[2], kernel_size=(3, 1), padding=(1, 0))
        # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(3, 3), padding=1)
        self.p3_31 = nn.Conv2d(c3[1], c3[2], kernel_size=(1, 3), padding=(0, 1))
        self.p3_32 = nn.Conv2d(c3[1], c3[3], kernel_size=(3, 1), padding=(1, 0))
        # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_1(x))
        p21 = F.relu(self.p2_21(p2))
        p22 = F.relu(self.p2_22(p2))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p31 = F.relu(self.p3_31(p3))
        p32 = F.relu(self.p3_32(p3))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        return torch.cat((p1, p21, p22, p31, p32, p4), dim=1)


class GoogLeNet(BasicNN):

    required_shape = (224, 224)

    def __init__(self, in_channels, out_features,
                 version='1', regression=False, dropout_rate=0.,
                 **kwargs):
        """经典GoogLeNet模型。

        :param in_channels: 输入通道
        :param out_features: 输出特征
        :param device: 设置本网络所处设备
        """
        supported = {'1', '2'}
        if version == '1':
            get_blocks = self.__get_version1
            multi_in = 1024
        elif version == '2':
            get_blocks = self.__get_version2
            multi_in = 2048
        else:
            raise NotImplementedError(f'暂不支持的GoogLeNet类型{version}，当前支持的类型包括{supported}')
        super().__init__(
            *get_blocks(in_channels),
            MultiOutputLayer(multi_in, out_features, init_meth=kwargs['init_meth']) if isinstance(out_features, Iterable)
            else nn.Sequential(*linear_output(multi_in, out_features, softmax=not regression)),
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
    def __get_version2(in_channels):
        b1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        b2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3), nn.ReLU(),
            nn.Conv2d(80, 192, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(192, 288, kernel_size=3, padding=1), nn.ReLU(),
        )
        # 以下参数为自创
        b3 = nn.Sequential(
            Inception_v2A(288, 128, (128, 192), (16, 32, 96), 64),
            Inception_v2A(480, 192, (96, 208), (16, 32, 48), 64),
            Inception_v2A(512, 256, (128, 256), (32, 64, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 形状降维成17x17
        )

        b4 = nn.Sequential(
            Inception_v2B(768, 256, (80, 160, 320), (16, 32, 64, 96, 128), 128),
            Inception_v2B(832, 256, (80, 160, 320), (16, 32, 64, 96, 128), 128),
            Inception_v2B(832, 384, (96, 192, 384), (24, 48, 80, 96, 128), 128),
            Inception_v2B(1024, 384, (96, 192, 384), (24, 48, 80, 96, 128), 128),
            Inception_v2B(1024, 384, (128, 256, 512), (48, 96, 128, 192, 256), 128),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 形状降维成7x7
        )

        b5 = nn.Sequential(
            Inception_v2C(1280, 512, (128, 256, 256), (48, 96, 128, 128), 256),
            Inception_v2C(1536, 768, (160, 320, 320), (48, 96, 128, 128), 384),  # 总通道数为2048
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return b1, b2, b3, b4, b5

    @staticmethod
    def get_required_shape(version='1'):
        supported = ['1', '2']
        if version == '1':
            return (224, 224)
        elif version == '2':
            return (299, 299)
        else:
            raise NotImplementedError(f'不支持的版本{version}，目前支持{supported}')

