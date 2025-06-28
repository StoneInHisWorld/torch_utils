from collections import OrderedDict

import torch
from torch import nn

from layers.add_positionEmbeddings import AddPositionEmbs
from networks.basic_nn import BasicNN
from networks.nets.resnet import ResNet50

supported_models = ['r50_b16', 'b16']
supported_classifier = ['token', 'token_unpooled', 'gap', 'unpooled']
supported_resnets = ['resnet50', None]


class EncoderLayer(nn.Module):

    def __init__(self, classifier,
                 num_hiddens, emb_size, num_telayer_head,
                 ffn_num_hiddens, num_telayer, dropout_rate,
                 add_pos_emb=True
                 ):
        # 编码器
        assert classifier in supported_classifier, f"无效的编码器分类符{classifier}，支持的分类符有{supported_classifier}"
        super(EncoderLayer, self).__init__()
        if add_pos_emb:
            # 加上分类符
            self.position_embedding = AddPositionEmbs('bert', (emb_size + 1, num_hiddens))
        else:
            self.position_embedding = nn.Identity()
        self.classifier = classifier
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                num_hiddens, num_telayer_head, ffn_num_hiddens, batch_first=True
            ),
            num_layers=num_telayer,
            norm=nn.LayerNorm(num_hiddens)
        )
        self.dropout = nn.Dropout(dropout_rate)
        # 加上分类符
        self.norm_layer = nn.LayerNorm([emb_size + 1, num_hiddens])

    def forward(self, inputs):
        if len(inputs.shape) == 4:
            bs, c, h, w = inputs.shape
            # 将序列长度移动到第1维，通道维移动到第2维
            inputs = inputs.permute(0, 2, 3, 1)
            inputs = inputs.reshape([bs, h * w, c])
        elif len(inputs.shape) == 3:
            bs, h_w, c = inputs.shape
            # # 将序列长度移动到第1维，通道维移动到第2维
            # inputs = inputs.permute(0, 2, 3, 1)
            # inputs = inputs.reshape([bs, h * w, c])
        else:
            raise ValueError(f"支持的输入形状是三维或四维，但得到的输入形状是{len(inputs.shape)}的！")

        # 在此添加类别符
        if self.classifier in ['token', 'token_unpooled']:
            cls = torch.zeros(1, 1, c).repeat(bs, 1, 1)
            inputs = torch.cat([cls, inputs], 1)
        inputs = self.position_embedding(inputs)
        inputs = self.dropout(inputs)
        inputs = self.encoder(inputs)
        return self.norm_layer(inputs)


class FetchClass(nn.Module):

    def __init__(self, which):
        super(FetchClass, self).__init__()
        # 编码器
        assert which in supported_classifier, f"无效的编码器分类符{which}，支持的分类符有{supported_classifier}"
        self.which = which

    def forward(self, inputs):
        if self.which == 'token':
            # 0号位即为分类标签
            inputs = inputs[:, 0]
        elif self.which == 'gap':
            inputs = torch.mean(inputs, dim=list(range(1, len(inputs.shape) - 1)))  # (1,) or (1,2)
        return inputs


class VisionTransformer(BasicNN):
    """
    VisionTransformer标准实现
    根据-An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale-
    https://doi.org/10.48550/arXiv.2010.11929 提供的架构编写
    """

    def __init__(self, which, in_channels, num_classes: int = None, **kwargs):
        # len_pic_seq：图片块序列长度
        # num_hiddens：词汇表中每个词汇长度
        # 将图片切块
        self.input_size = (in_channels, 224, 224) if 'input_size' not in kwargs.keys() \
            else kwargs.pop('input_size')
        self.__get_model_config(which)
        # 获取ResNet层
        resnet, output_channels = self.find_a_resnet(in_channels)
        linear_projector = nn.Conv2d(
            output_channels, self.hidden_size,
            kernel_size=self.patches_size, stride=self.patches_size
        )
        # 获取编码器
        encoder = EncoderLayer(
            self.classifier, self.hidden_size, self.tran_emb_size,
            self.tran_num_heads, self.tran_mlp_dim, self.tran_num_layers,
            self.tran_dropout_rate
        )
        fetch_class = FetchClass(self.classifier)
        # ？
        pre_logits = nn.Identity() if self.representation_size is None else \
            nn.Sequential(
                nn.Linear(self.hidden_size, self.representation_size), nn.Tanh()
            )
        # 感知机头
        if num_classes:
            mlp_head = nn.Linear(
                self.hidden_size if self.representation_size is None
                else self.representation_size,
                num_classes
            )
        else:
            mlp_head = nn.Identity()
        kwargs = {'input_size': self.input_size, **kwargs}
        super(VisionTransformer, self).__init__(OrderedDict([
            ('resnet', resnet), ('linear_projector', linear_projector),
            ('encoder', encoder), ('fetch_class', fetch_class),
            ('pre_logits', pre_logits), ('mlp_head', mlp_head)
        ]), **kwargs)
        # # 给父类对象赋值input_size
        # self.input_size = (in_channels, 224, 224) if 'input_size' not in kwargs.keys() \
        #     else kwargs['input_size']

    def find_a_resnet(self, in_channels):
        """根据参数指定来分配ResNet块
        :return ResNet网络，输出通道数
        """
        if self.resnet_type == 'resnet50':
            resnet = ResNet50(in_channels, [3, 4, 9])
            del resnet.avgpool, resnet.flatten, resnet.output
            output_channels = resnet.in_channels
        else:
            resnet = nn.Identity()
            output_channels = in_channels
        return resnet, output_channels

    def __get_model_config(self, which):
        assert which in supported_models, f'不支持的模型类型{which}，支持的模型类型包括{supported_models}'
        if which == 'r50_b16':
            self.__get_r50_b16_config()
        elif which == 'b16':
            self.__get_b16_config()
        else:
            pass

    def __get_b16_config(self):
        """指定ViT_B16的配置"""
        self.resnet_type = None
        self.patches_size = 16
        self.hidden_size = 768
        self.tran_emb_size = (self.input_size[1] // self.patches_size) * (self.input_size[2] // self.patches_size)
        self.tran_mlp_dim = 3072
        self.tran_num_heads = 12
        self.tran_num_layers = 12
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.0
        self.classifier = 'token'
        self.representation_size = None

    def __get_r50_b16_config(self):
        """Returns the Resnet50 + ViT-B/16 configuration."""
        self.__get_b16_config()
        self.resnet_type = 'resnet50'
        self.tran_attention_dropout_rate = 0.1
        self.patches_size = 1
        self.resnet_num_layers = (3, 4, 9)
        self.resnet_width_factor = 1

#
# inputs = torch.rand([4, 3, 224, 224])
# transformer = VisionTransformer('b16', 3, num_classes=784)
# inputs = transformer(inputs)
# print(inputs.shape)
