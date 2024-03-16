from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.multi_output import MultiOutputLayer


class AdaWZYNet(BasicNN):

    required_shape = (256, 256)
    down_sampling = lambda i, k: nn.Sequential(
        # 下采样
        nn.Conv2d(i, i, kernel_size=k, stride=2, padding=1),
        nn.LeakyReLU()
    )
    fea_extract_and_down_sampling = lambda i, o, k, bm, dp: nn.Sequential(
        # 提取特征+下采样
        nn.Conv2d(i, o, kernel_size=k, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(o, momentum=bm),
        nn.Dropout(dp)
    )
    output_path = lambda: []

    def __init__(self,
                 input_channels,
                 input_shape, compressed_shape, out_features,
                 base_channels=16,
                 kernel_size=3, bn_momen=0.95, dropout_rate=0.,
                 **kwargs):
        """通过不断卷积、下采样，提取图片信息的卷积神经网络。
        本改编版本可根据输入形状进行网络结构的自适应。

        参考：

        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023
        :param in_channels: 输入特征通道数。
        :param out_features: 输出数据通道数。
        :param kwargs: BasicNN关键词参数。
        :param base_channels: 基础通道，数值越高意味着网络越复杂。
        :param out_features: 输出列数。若指定为整数列表，则最终输出层会转化为对应数目的多路径输出层，所有输出层同时预测，结果拼接在一起输出。
        :param kernel_size: 卷积层的感受野大小。请慎重选择Kernel_size，可能会导致数形不匹配问题！
        :param bn_momen: BatchNorm层的动量参数。
        :param dropout_rate: 是否进行Dropout。取值需为0~1，表示每个Dropout层的抛弃比例。若指定为0，则不进行Dropout。
        """
        layers = [
            nn.BatchNorm2d(input_channels, momentum=bn_momen),
            nn.Conv2d(input_channels, base_channels, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU()
        ]
        # 构造基本结构
        i = 1
        cur_in, cur_out, cur_shape = base_channels * i, base_channels * (i + 1), input_shape // 2
        while cur_shape > compressed_shape:
            layers.append(AdaWZYNet.fea_extract_and_down_sampling(
                cur_in, cur_out, kernel_size, bn_momen, dropout_rate
            ))
            i, cur_in, cur_out = i + 1, cur_out, base_channels * (i + 2)
            cur_shape //= 2
        # 构造输出通道
        init_meth = kwargs['init_meth'] if ('init_meth' in kwargs.keys()
                                            and kwargs['init_meth'] != 'state') \
            else 'normal'
        layers.append(nn.Sequential(
            nn.MaxPool2d(compressed_shape),
            MultiOutputLayer(
                cur_in, out_features,
                dropout_rate=dropout_rate, momentum=bn_momen, init_meth=init_meth,
            )  # cur_in保存着最后的卷积层的输出通道
        ))
        super().__init__(*layers, **kwargs)
