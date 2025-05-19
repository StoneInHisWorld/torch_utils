from collections import OrderedDict

from torch import nn

from layers.resnet_blocks import Bottleneck
from networks import BasicNN


def ResNet50(in_channels,
             num_layers=None, width=64,
             # norm_layer=nn.BatchNorm2d, norm_kwargs=None,
             num_classes=1000):
    """
    ResNet50标准实现
    根据-Deep Residual Learning for Image Recognition-
    https://arxiv.org/abs/1512.03385提供的架构编写
    """
    # if norm_kwargs is None:
    #     norm_kwargs = {}
    if num_layers is None:
        num_layers = [3, 4, 6, 3]
    return ResNet(
        Bottleneck, num_layers, in_channels,
        width=width,
        # norm_layer=norm_layer,  norm_kwargs=norm_kwargs,
        num_classes=num_classes
    )


class ResNet(BasicNN):

    def __init__(self,
                 block_type, num_layers, in_channels,
                 width=64, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 num_classes=1000,
                 *args, **kwargs):
        """
        ResNet标准实现
        根据-Deep Residual Learning for Image Recognition-
        https://arxiv.org/abs/1512.03385提供的架构编写
        """
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_channels = width  # 初始通道数
        # Root Block
        root_block = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # ResNet Stages
        # resnet_stages = self._make_layer(block, width, layers[0], stride=1)
        resnet_stages = nn.Sequential(OrderedDict([
            ('stage0', self._make_layer(block_type, width, num_layers[0], norm_layer, norm_kwargs))
        ] + [
            (f'stage{i}', self._make_layer(block_type, width * 2 ** i, nl, norm_layer, norm_kwargs, stride=2))
            for i, nl in enumerate(num_layers[1:], 1)
        ])
            # ,
            # *[self._make_layer(block, width * 2 ** (i+1), nl, stride=2)
            #   for i, nl in enumerate(num_layers[1:])]
            # self._make_layer(block, width * 2, layers[1], stride=2),
            # self._make_layer(block, width * 4, layers[2], stride=2),
            # self._make_layer(block, width * 8, layers[3], stride=2),
        )
        # self.layer1 = self._make_layer(block, width, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, width*2, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, width*4, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, width*8, layers[3], stride=2)

        # 分类头
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output = nn.Linear(self.in_channels, num_classes)
        super(ResNet, self).__init__(OrderedDict([
            ('root_block', root_block), ('resnet_stages', resnet_stages),
            ('avgpool', avgpool), ("flatten", nn.Flatten(1)), ('output', output)
        ]), *args, **kwargs)

    # def forward(self, x):
    #     # x = self.conv1(x)
    #     # x = self.gn1(x)
    #     # x = self.relu(x)
    #     # x = self.maxpool(x)
    #     #
    #     # x = self.layer1(x)
    #     # x = self.layer2(x)
    #     # x = self.layer3(x)
    #     # x = self.layer4(x)
    #     #
    #     # x = self.avgpool(x)
    #     # x = torch.flatten(x, 1)
    #     # x = self.fc(x)
    #     # return x
    #     x = self.root_block(x)
    #     x = self.resnet_stages(x)
    #     x = self.avgpool(x)
    #     return self.output(torch.flatten(x, 1))

    def _make_layer(self, block_type, planes, num_blocks, norm_layer, norm_kwargs, stride=1):
        downsample = None
        # 当 stride≠1 或通道数变化时，需要下采样
        if stride != 1 or self.in_channels != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.GroupNorm(32, planes * block_type.expansion),  # 与残差块一致
            )

        layers = []
        layers.append(block_type(
            self.in_channels, planes, stride, downsample
            # , norm_layer, norm_kwargs
        ))
        self.in_channels = planes * block_type.expansion  # 更新输入通道数
        for _ in range(1, num_blocks):
            layers.append(block_type(self.in_channels, planes,
                                     # norm_layer=norm_layer, norm_kwargs=norm_kwargs
                                     ))

        return nn.Sequential(*layers)


# t = torch.randn(4, 3, 224, 224)
# model = ResNet50(3, [3, 4, 9])
# t = model(t)
# pass