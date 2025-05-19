from torch import nn


class ResnetBlock(nn.Module):
    """定义一个ResNet块"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化ResNet块。

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构造一个卷积块

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('不支持的padding模式[%s]' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('不支持的padding模式[%s]' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """使用跳过链接的前向传播"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Bottleneck(nn.Module):
    """用于ResNet-50/101/152的ResNet-Stage"""
    expansion = 4  # 每个块输出通道数是中间层的 4 倍

    def __init__(self,
                 in_channels, mid_channels,
                 stride=1, downsample=None,
                 # norm_layer=nn.BatchNorm2d, norm_kwargs=None
                 ):
        super(Bottleneck, self).__init__()
        # if norm_kwargs is None:
        #     norm_kwargs = {'num_features': mid_channels}
        # else:
        #     norm_kwargs['num_features'] = mid_channels
        main_branch = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels * self.expansion),
        ]
        # norm_kwargs['num_features'] = mid_channels * self.expansion
        # main_branch.append(norm_layer(mid_channels * self.expansion))
        self.main_branch = nn.Sequential(*main_branch)
        self.downsample = downsample
        self.output = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        # self.gn1 = nn.GroupNorm(groups, mid_channels)
        # self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.gn2 = nn.GroupNorm(groups, mid_channels)
        # self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        # self.gn3 = nn.GroupNorm(groups, mid_channels * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample

    def forward(self, inputs):
        # identity = inputs
        #
        # out = self.conv1(inputs)
        # out = self.gn1(out)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.gn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.gn3(out)
        #
        # if self.downsample is not None:
        #     identity = self.downsample(inputs)
        #
        # out += identity
        # out = self.relu(out)
        #
        # return out
        output = self.main_branch(inputs)
        if self.downsample is not None:
            inputs = self.downsample(inputs)
        output += inputs
        return self.output(output)

