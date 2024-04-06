from abc import abstractmethod
from typing import Tuple, List, Sized

import torch
from torch import nn
from torch.nn import functional as F


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
        :param c1: path1每层的输出通道，len(c1) == 1。线路1，单1x1卷积层，不改变形状
        :param c2: path2每层的输出通道，len(c2) == 2。线路2，1x1卷积层后接3x3卷积层，不改变形状
        :param c3: path3每层的输出通道，len(c3) == 3。线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        :param c4: path4每层的输出通道，len(c4) == 1。线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        """
        super(Inception_v2A, self).__init__(in_channels, c1, c2, c3, c4)

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
        # 线路2，1x1卷积层后接3x3卷积层，不改变形状
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


class Inception_v2B(Inception):

    def __init__(self, in_channels, c1, c2, c3, c4, n=7):
        """构成GoogLeNet的Inception块，版本2。结构来自于论文中的图6

        参考：

        [1]https://zhuanlan.zhihu.com/p/194382937. 2020.08.22/2024.03.20

        [2] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中每层的通道数，len(c1) == 1。线路1，单1x1卷积层，不改变形状。
        :param c2: path2中每层的通道数，len(c2) == 3。线路2，1x1卷积层后接1xn、nx1卷积层，不改变形状。
        :param c3: path3中每层的通道数，len(c3) == 5。线路3，1x1卷积层后接两组1xn、nx1卷积层，不改变形状
        :param c4: path4中每层的通道数，len(c4) == 1。线路4，3x3平均汇聚层后接1x1卷积层，不改变形状。
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

    def get_paths(self, in_channels, *args):
        c1, c2, c3, c4, n = args
        # 线路1，单1x1卷积层，不改变形状
        p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
        self.p1 = p1[0]
        # 线路2，1x1卷积层后接1xn、nx1卷积层，不改变形状
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c2[1], c2[2], kernel_size=(n, 1), padding=(n // 2, 0))
        ]
        self.p2_1, self.p2_2, self.p2_3 = p2
        # 线路3，1x1卷积层后接两组1xn、nx1卷积层，不改变形状
        p3 = [
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.Conv2d(c3[0], c3[1], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c3[1], c3[2], kernel_size=(n, 1), padding=(n // 2, 0)),
            nn.Conv2d(c3[2], c3[3], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c3[3], c3[4], kernel_size=(n, 1), padding=(n // 2, 0))
        ]
        self.p3_1, self.p3_2, self.p3_3, self.p3_4, self.p3_5 = p3
        # 线路4，3x3平均汇聚层后接1x1卷积层，不改变形状
        p4 = [
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
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
        :param c1: path1中每层的输出通道，len(c1) == 1。线路1，单1x1卷积层，不改变形状。
        :param c2: path2中每层的输出通道，len(c2) == 3。线路2，1x1卷积层后接1x3、3x1卷积层，不改变形状.
        :param c3: path3中每层的输出通道，len(c3) == 4。线路3，1x1卷积层后接一个3x3卷积层，一组1x3、3x1卷积层，不改变形状。
        :param c4: path4中每层的输出通道，len(c4) == 1。线路4，3x3最大汇聚层后接1x1卷积层，不改变形状。
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
        # 线路2，1x1卷积层后接1x3、3x1卷积层，不改变形状
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_21 = nn.Conv2d(c2[0], c2[1], kernel_size=(1, 3), padding=(0, 1))
        self.p2_22 = nn.Conv2d(c2[0], c2[2], kernel_size=(3, 1), padding=(1, 0))
        # 线路3，1x1卷积层后接一个3x3卷积层，一组1x3、3x1卷积层，不改变形状
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(3, 3), padding=1)
        self.p3_31 = nn.Conv2d(c3[1], c3[2], kernel_size=(1, 3), padding=(0, 1))
        self.p3_32 = nn.Conv2d(c3[1], c3[3], kernel_size=(3, 1), padding=(1, 0))
        # 线路4，3x3最大汇聚层后接1x1卷积层，不改变形状
        self.p4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
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


class Inception_v3A(Inception):

    def __init__(self, in_channels, c1, c2, c3, c4):
        """构成GoogLeNet的Inception块，版本3，结构来自于论文提供的model.txt

        参考：

        [1] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中每层的输出层，len(c1) == 1。线路1，单1x1卷积层，不改变形状
        :param c2: path2中每层的输出层，len(c2) == 2。线路2，1x1卷积层后接5x5卷积层，不改变形状
        :param c3: path3中每层的输出层，len(c3) == 3。线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        :param c4: path4中每层的输出层，len(c4) == 1。线路4，3x3平均汇聚层后接1x1卷积层，不改变形状
        """
        super(Inception_v3A, self).__init__(in_channels, c1, c2, c3, c4)

    def check_para(self, *args):
        assert len(args) == 4, f'需要为{self.__class__.__name__}提供4个通道参数，但是收到了{len(args)}'
        c1, c2, c3, c4 = args
        assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
        assert len(c2) == 2, f'第二条路径的输出通道数c2只能指定2个，然而收到了{len(c2)}个！'
        assert len(c3) == 3, f'第三条路径的输出通道数c3只能指定3个，然而收到了{len(c3)}个！'
        assert not isinstance(c4, Sized), f'第四条路径的输出通道数c4只能为数字！'

    def get_paths(self, in_channels, *args):
        c1, c2, c3, c4 = args
        # 线路1，单1x1卷积层，不改变形状
        p1 = [nn.Conv2d(in_channels, c1, kernel_size=1)]
        self.p1 = p1[0]
        # 线路2，1x1卷积层后接5x5卷积层，不改变形状
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=5, padding=2)
        ]
        self.p2_1, self.p2_2 = p2
        # 线路3，1x1卷积层后接两个3x3卷积层，不改变形状
        p3 = [
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
            nn.Conv2d(c3[1], c3[2], kernel_size=3, padding=1)
        ]
        self.p3_1, self.p3_2, self.p3_3 = p3
        # 线路4，3x3平均汇聚层后接1x1卷积层，不改变形状
        p4 = [
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1)
        ]
        self.p4_1, self.p4_2 = p4
        return p1, p2, p3, p4


class Inception_v3B(Inception):

    def __init__(self, in_channels, c1, c2):
        """构成GoogLeNet的Inception块，版本3，结构来自于论文提供的model.txt。
        用于第一段Inception块后的下采样。包含三个分支：
        线路1，单3x3卷积层，形状减半；线路2，1x1卷积层后接两个3x3卷积层，形状减半；线路3，单3x3最大汇聚层，形状减半。

        参考：

        [1] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中每层的输出层，len(c1) == 1。线路1，单3x3卷积层，形状减半。
        :param c2: path2中每层的输出层，len(c2) == 3。线路2，1x1卷积层后接两个3x3卷积层，形状减半。
        """
        super(Inception_v3B, self).__init__(in_channels, c1, c2)

    def check_para(self, *args):
        assert len(args) == 2, f'需要为{self.__class__.__name__}提供2个通道参数，但是收到了{len(args)}'
        c1, c2 = args
        assert not isinstance(c1, Sized), f'第一条路径的输出通道数c1只能为数字！'
        assert len(c2) == 3, f'第二条路径的输出通道数c2只能指定3个，然而收到了{len(c2)}个！'

    def get_paths(self, in_channels, *args):
        c1, c2 = args
        # 线路1，单3x3卷积层，形状减半
        p1 = [nn.Conv2d(in_channels, c1, kernel_size=3, stride=2)]
        self.p1 = p1[0]
        # 线路2，1x1卷积层后接两个3x3卷积层，形状减半
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.Conv2d(c2[1], c2[2], kernel_size=3, stride=2)
        ]
        self.p2_1, self.p2_2, self.p2_3 = p2
        # 线路3，单3x3最大汇聚层，形状减半
        p3 = [nn.MaxPool2d(kernel_size=3, stride=2)]
        self.p3 = p3[0]
        return p1, p2, p3


class Inception_v3C(Inception):

    def __init__(self, in_channels, c1, c2, n=7):
        """构成GoogLeNet的Inception块，版本3，结构来自于论文提供的model.txt。
        用于第二段Inception块后的下采样。包含三个分支：
        线路1，1x1卷积层后接3x3卷积层，形状减半；
        线路2，1x1卷积层后接一组1xn、nx1卷积层，最后一组3x3卷积层使形状减半；
        线路3，单3x3最大汇聚层，形状减半。

        参考：

        [1] Christian Szegedy, Vincent Vanhoucke,  Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna.
        Rethinking the Inception Architecture for Computer Vision[J]
        https://arxiv.org/abs/1512.00567.
        :param in_channels: 输入通道
        :param c1: path1中每层的输出层，len(c1) == 2。线路1，1x1卷积层后接3x3卷积层，形状减半。
        :param c2: path2中每层的输出层，len(c2) == 3。线路2，1x1卷积层后接一组1xn、nx1卷积层，最后一组3x3卷积层使形状减半。
        :param n: path2中分解的层的感受野，n最好为大于0的奇数
        """
        super(Inception_v3C, self).__init__(in_channels, c1, c2, n)

    def check_para(self, *args):
        assert len(args) == 3, f'需要为{self.__class__.__name__}提供3个通道参数，但是收到了{len(args)}'
        c1, c2, n = args
        assert len(c1) == 2, f'第一条路径的输出通道数c1只能指定2个，然而收到了{len(c1)}个！'
        assert len(c2) == 4, f'第二条路径的输出通道数c2只能指定4个，然而收到了{len(c2)}个！'
        assert n > 0, f'n值应该大于0，但是收到了{n}！'

    def get_paths(self, in_channels, *args):
        c1, c2, n = args
        # 线路1，1x1卷积层后接3x3卷积层，形状减半
        p1 = [
            nn.Conv2d(in_channels, c1[0], kernel_size=1),
            nn.Conv2d(c1[0], c1[1], kernel_size=3, stride=2)
        ]
        self.p1_1, self.p1_2 = p1
        # 线路2，1x1卷积层后接一组1xn、nx1卷积层，最后一组3x3卷积层使形状减半
        p2 = [
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.Conv2d(c2[0], c2[1], kernel_size=(1, n), padding=(0, n // 2)),
            nn.Conv2d(c2[1], c2[2], kernel_size=(n, 1), padding=(n // 2, 0)),
            nn.Conv2d(c2[2], c2[3], kernel_size=3, stride=2)
        ]
        self.p2_1, self.p2_2, self.p2_3, self.p2_4 = p2
        # 线路3，单3x3最大汇聚层，形状减半
        p3 = [nn.MaxPool2d(kernel_size=3, stride=2)]
        self.p3 = p3[0]
        return p1, p2, p3
