from torch import nn

from networks.basic_nn import BasicNN
from networks.layers.multi_output import MultiOutputLayer


class WZYNetEssay(BasicNN):

    required_shape = (256, 256)

    def __init__(self, input_channels, base_channels, out_features,
                 kernel_size=3, bn_momen=0.95, dropout_rate=0.,
                 **kwargs):
        """
        通过不断卷积，下采样，提取图片信息的网络。
        参考：
        [1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023
        :param in_channels: 输入特征通道数。
        :param out_features: 输出数据通道数。
        :param kwargs: BasicNN关键词参数。
        :param base_channels: 基础通道，数值越高意味着网络越复杂。
        :param out_features: 输出列数。若指定为整数列表，则最终输出层会转化为对应数目的多路径输出层，所有输出层同时预测，结果拼接在一起输出。
        :param kernel_size: 卷积层的感受野大小。
        :param bn_momen: BatchNorm层的动量参数。
        :param dropout_rate: 是否进行Dropout。取值需为0~1，表示每个Dropout层的抛弃比例。若指定为0，则不进行Dropout。
        """
        layers = [
            nn.BatchNorm2d(input_channels, momentum=bn_momen),
            nn.Conv2d(input_channels, base_channels, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels, momentum=bn_momen),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels * 2, momentum=bn_momen), nn.Dropout(0.3),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels * 2, momentum=bn_momen),
            nn.Conv2d(base_channels * 2, base_channels * 3, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels * 3, momentum=bn_momen), nn.Dropout(0.3),
            nn.Conv2d(base_channels * 3, base_channels * 3, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels * 3, momentum=bn_momen),
            nn.Conv2d(base_channels * 3, base_channels * 4, kernel_size=kernel_size,
                      stride=2, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels * 4, momentum=bn_momen), nn.MaxPool2d(2),
            MultiOutputLayer(
                base_channels * 4, out_features, init_meth=kwargs['init_meth'], dropout_rate=dropout_rate,
                momentum=bn_momen,
            ),
        ]
        super().__init__(*layers, **kwargs)
