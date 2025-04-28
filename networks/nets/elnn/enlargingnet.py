import torch
from torch.nn import Upsample, ReLU, BatchNorm2d

from utils.func.pytools import check_para
from .__el_block import ELBlockGenerator
from .__output_block import OutputBlockGenerator
from networks import BasicNN


version_supported = ['v1', 'v2']


class EnLargingNN(BasicNN):

    # activation = ReLU
    # norm_layer = BatchNorm2d

    def __init__(self,
                 fea_chan, out_chan, input_shape, output_shape,
                 base=2, exp=2, version='v1', norm_layer=BatchNorm2d, activation=ReLU,
                 **kwargs):
        """请根据需要指定exp参数，上采样的次数等于exp的次数

        :param fea_chan:
        :param out_chan:
        :param output_shape:
        :param base:
        :param exp:
        :param kwargs:
        """
        # 版本检查
        assert check_para('version', version, version_supported), f'不支持的version参数{version}！'
        self.version = version
        # 层设置
        self.norm_layer = norm_layer
        self.activation = activation
        fea_chan = int(fea_chan)
        o_fea_chan_array = [int(base ** i) for i in range(int(exp + 1))]
        layers = []
        cur_shape = input_shape
        # 构造块生成器
        og = OutputBlockGenerator(version, norm_layer, activation)
        elg = ELBlockGenerator(version, norm_layer, activation)
        # 上采样
        for o in o_fea_chan_array:
            cur_shape, elbks = elg.get_blocks(fea_chan, o, cur_shape, output_shape)
            # cur_shape, elbks = self.get_ELBlock(fea_chan, o, cur_shape, output_shape)
            layers += elbks
            fea_chan = o
        layers.append(Upsample(output_shape))
        # 使用三层卷积压缩通道以调整至输出通道
        for o in torch.logspace(
            exp, torch.log(torch.tensor(out_chan)) / torch.log(torch.tensor(base)), 3, base=base
        ):
            o = int(o)
            layers += og.get_blocks(fea_chan, o)
            # layers.append(OutputBlock(fea_chan, o, version, norm_layer, activation))
            fea_chan = o
        # 去掉末尾的标准化层和激活函数
        layers.pop()
        layers.pop()
        super().__init__(*layers, **kwargs)

    # def get_ELBlock(self, i, o, cur_shape, required_shape):
    #     ct_s, ct_k, ct_p = 2, 3, 1
    #     oshape_h = (cur_shape[0] - 1) * ct_s + ct_k - 2 * ct_p
    #     oshape_w = (cur_shape[0] - 1) * ct_s + ct_k - 2 * ct_p
    #     if oshape_h < required_shape[0] and oshape_w < required_shape[1]:
    #         return (cur_shape[0] * 2 - 1, cur_shape[1] * 2 - 1), [
    #             # 提取特征
    #             Conv2d(i, o, kernel_size=3, stride=1, padding=1),
    #             self.norm_layer(o),
    #             self.activation(),
    #             # 上采样2倍
    #             ConvTranspose2d(o, o, kernel_size=ct_k, stride=ct_s, padding=ct_p),
    #             self.norm_layer(o),
    #             self.activation()
    #         ]
    #     else:
    #         return cur_shape, [
    #             # 提取特征
    #             Conv2d(i, o, kernel_size=3, stride=1, padding=1),
    #             self.norm_layer(o),
    #             self.activation(),
    #         ]
    #
    # def get_OutputBlock(self, i, o):
    #     if elnn.version == 'v1':
    #         return [
    #             # 压缩特征
    #             Conv2d(i, o, kernel_size=3, stride=1, padding=1),
    #             self.norm_layer(o),
    #             self.activation(),
    #         ]
    #     elif elnn.version == 'v2':
    #         return [
    #             # 压缩特征
    #             Conv2d(i, o, kernel_size=1),
    #             self.norm_layer(o),
    #             self.activation(),
    #         ]
    #     else:
    #         raise ValueError(f"不支持的版本{elnn.version}")
