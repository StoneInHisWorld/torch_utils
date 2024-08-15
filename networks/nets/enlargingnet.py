import torch

from networks import BasicNN
from torch.nn import Conv2d, ReLU, BatchNorm2d, ConvTranspose2d, Upsample, Linear
from utils.func.pytools import check_para


class EnLargingNN(BasicNN):

    activation = ReLU
    norm_layer = BatchNorm2d
    version_supported = ['1', '2']

    def __init__(self,
                 fea_chan, out_chan, required_shape,
                 base=2, exp=2, version='1',
                 **kwargs):
        """请根据需要指定exp参数，上采样的次数等于exp的次数，上采样的结果形状有可能会超过required_shape

        :param fea_chan:
        :param out_chan:
        :param required_shape:
        :param base:
        :param exp:
        :param kwargs:
        """
        fea_chan = int(fea_chan)
        o_fea_chan_array = [int(base ** i) for i in range(int(exp + 1))]
        layers = []
        assert check_para('version', version, EnLargingNN.version_supported), 'version参数设置错误！'
        self.version = version
        # 上采样
        for o in o_fea_chan_array:
            layers += self.get_ELBlock(fea_chan, o)
            fea_chan = o
        layers.append(Upsample(required_shape))
        # 使用三层卷积压缩通道以调整至输出通道
        for o in torch.logspace(
            exp, torch.log(torch.tensor(out_chan)) / torch.log(torch.tensor(base)), 3, base=base
        ):
            o = int(o)
            layers += self.get_OutputBlock(fea_chan, o)
            fea_chan = o
        # for o in list(reversed(o_fea_chan_array))[:-1]:
        #     layers += self.get_OutputBlock(fea_chan, o)
        #     fea_chan = o
        # 去掉末尾的标准化层和激活函数
        # layers.append(Conv2d(fea_chan, out_chan, kernel_size=3, stride=1, padding=1))
        layers.pop()
        layers.pop()
        super().__init__(*layers, **kwargs)

    def get_ELBlock(self, i, o):
        return [
            # 提取特征
            Conv2d(i, o, kernel_size=3, stride=1, padding=1),
            self.norm_layer(o),
            self.activation(),
            # 上采样2倍
            ConvTranspose2d(o, o, kernel_size=3, stride=2, padding=1),
            self.norm_layer(o),
            self.activation()
        ]

    def get_OutputBlock(self, i, o):
        if self.version == '1':
            return [
                # 压缩特征
                Conv2d(i, o, kernel_size=3, stride=1, padding=1),
                self.norm_layer(o),
                self.activation(),
            ]
        elif self.version == '2':
            return [
                # 压缩特征
                Conv2d(i, o, kernel_size=1),
                self.norm_layer(o),
                self.activation(),
            ]
        else:
            raise ValueError(f"不支持的版本{self.version}")
