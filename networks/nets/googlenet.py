import torch
import torch.nn.functional as F
from torch import nn

from networks.layers.multi_output import MultiOutputLayer
from networks.basic_nn import BasicNN


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        """
        构成GoogLeNet的Inception块
        :param in_channels: 输入通道
        :param c1: path1中层的感受野，len(c1) == 1
        :param c2: path2中层的感受野，len(c2) == 2
        :param c3: path3中层的感受野，len(c3) == 2
        :param c4: path4中层的感受野，len(c4) == 2
        :param kwargs: 构建nn.Module对象的关键词参数
        """
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(BasicNN):

    def __init__(self, in_channels, out_features, init_meth='xavier', with_checkpoint=False,
                 device='cpu'):
        """
        构造经典GoogLeNet
        :param in_channels: 输入通道
        :param out_features: 输出特征
        :param device: 设置本网络所处设备
        """
        inception = Inception
        b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b3 = nn.Sequential(
            inception(192, 64, (96, 128), (16, 32), 32),
            inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b4 = nn.Sequential(
            inception(480, 192, (96, 208), (16, 48), 64),
            inception(512, 160, (112, 224), (24, 64), 64),
            inception(512, 128, (128, 256), (24, 64), 64),
            inception(512, 112, (144, 288), (32, 64), 64),
            inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b5 = nn.Sequential(
            inception(832, 256, (160, 320), (32, 128), 128),
            inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        super().__init__(
            device, init_meth, with_checkpoint,
            b1, b2, b3, b4, b5,
            # nn.Linear(1024, out_features),
            MultiOutputLayer(1024, out_features, init_meth=init_meth)
        )

