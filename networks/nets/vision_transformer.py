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
            self.position_embedding = AddPositionEmbs((emb_size + 1, num_hiddens))
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
        # if self.encoder is not None:
        #     bs, c, h, w = inputs.shape
        #     # 将序列长度移动到第1维，通道维移动到第2维
        #     inputs = inputs.reshape([bs, h * w, c])
        #
        #     # If we want to add a class token, add it here.
        #     if self.classifier in ['token', 'token_unpooled']:
        #         # cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        #         cls = torch.zeros(1, 1, c).repeat(bs, 1, 1)
        #         # cls = self.ini.tile(cls, [bs, 1, 1])
        #         inputs = torch.cat([cls, inputs], 1)
        #
        #     inputs = self.encoder(inputs)
        bs, c, h, w = inputs.shape
        # 将序列长度移动到第1维，通道维移动到第2维
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.reshape([bs, h * w, c])

        # 在此添加类别符
        if self.classifier in ['token', 'token_unpooled']:
            cls = torch.zeros(1, 1, c).repeat(bs, 1, 1)
            inputs = torch.cat([cls, inputs], 1)
        inputs = self.position_embedding(inputs)
        inputs = self.dropout(inputs)
        inputs = self.encoder(inputs)
        return self.norm_layer(inputs)

        # if self.classifier == 'token':
        #     # 0号位即为分类标签
        #     inputs = inputs[:, 0]
        # elif self.classifier == 'gap':
        #     inputs = torch.mean(inputs, dim=list(range(1, len(inputs.shape) - 1)))  # (1,) or (1,2)
        # return inputs


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


#
#
# class ResNet(BasicNN):
#
#     def __init__(self, which, in_channel, width):
#         assert which in supported_resnets, f'不支持的ResNet类型{which}！支持的类型包括{supported_resnets}'
#         # 输入通道需要根据实际情况调整（例如 RGB 图像为 3）
#         width = int(64 * width)
#         conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channel, width,
#                 kernel_size=7, stride=2, padding=3, bias=False
#             ),
#             nn.GroupNorm(64, in_channel),
#             nn.ReLU(),
#             nn.MaxPool2d(3, 2, 1)
#         )
#
#     def _make_stage(self, block, planes, blocks, stride=1):
#         downsample = None
#         # 当 stride≠1 或通道数变化时，需要下采样
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.inplanes,
#                     planes * block.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False
#                 ),
#                 nn.GroupNorm(32, planes * block.expansion),  # 与残差块一致
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion  # 更新输入通道数
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)


class VisionTransformer(BasicNN):
    """
    VisionTransformer标准实现
    根据-An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale-
    https://doi.org/10.48550/arXiv.2010.11929 提供的架构编写
    """

    def __init__(self,
                 which, in_channels,
                 # resnet=None, classifier='token',
                 # patch_size=16, patching='patching',
                 # num_telayer=12, num_telayer_head=12,
                 # ffn_num_hiddens=3072, representation_size=None,
                 num_classes=None, *args, **kwargs
                 ):
        # len_pic_seq：图片块序列长度
        # num_hiddens：词汇表中每个词汇长度
        # if patching == 'patching':
        #     patching = Patching2D(self.patch_size)
        # input_size = (3, 224, 224) if 'input_size' not in kwargs.keys() else kwargs['input_size']
        # input_channel, input_h, input_w = input_size
        # # len_pic_seq = input_h // patch_size * (input_w // patch_size)
        # # 获取resnet层
        # resnet, output_channels, patch_size = self.find_a_resnet(resnet, input_channel)
        # assert resnet in supported_resnets, f'不支持的ResNet类型{resnet}！支持的类型包括{supported_resnets}'
        # if resnet == 'resnet50':
        #     # resnet_width = int(64 * resnet_width)
        #     resnet = torchvision.models.resnet50(num_classes=input_channel)
        #     # resnet = nn.Sequential(
        #     #     nn.Conv2d(input_channel, resnet_width, 7, 2, bias=False),
        #     #     nn.GroupNorm(64, resnet_width),
        #     #     nn.ReLU(),
        #     #     nn.MaxPool2d(3, 2, )
        #     # )
        # else:
        #     resnet = nn.Identity()
        # 线性投影层
        # num_hiddens = input_channel * patch_size ** 2

        # 将图片切块
        input_size = (3, 224, 224) if 'input_size' not in kwargs.keys() else kwargs['input_size']
        # input_channel, _, _ = input_size
        self.__get_model_config(which, input_size)
        # 获取ResNet层
        resnet, output_channels = self.find_a_resnet(in_channels)
        linear_projector = nn.Conv2d(
            output_channels, self.hidden_size,
            kernel_size=self.patches_size, stride=self.patches_size
        )
        # # 嵌入层
        # embedding = nn.Embedding(len_pic_seq, num_hiddens)
        # 设置分类符号
        # self.classifier = classifier
        # if classifier in ['token', 'token_unpooled']:
        #     self.init_cls = torch.zeros(1, 1, input_channel)
        # embed_size = (input_size[1] // self.patches_size) * (input_size[2] // self.patches_size)
        encoder = EncoderLayer(
            self.classifier, self.hidden_size, self.tran_emb_size,
            self.tran_num_heads, self.tran_mlp_dim, self.tran_num_layers,
            self.tran_dropout_rate
        )
        fetch_class = FetchClass(self.classifier)
        # # 编码器
        # encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         num_hiddens, num_telayer_head, ffn_num_hiddens, batch_first=True
        #     ), num_layers=num_telayer,
        #     norm=nn.LayerNorm(num_hiddens)
        # )
        # ？
        pre_logits = nn.Identity() if self.representation_size is None else \
            nn.Sequential(
                nn.Linear(self.hidden_size, self.representation_size), nn.Tanh()
            )
        # 感知机头
        if num_classes:
            mlp_head = nn.Linear(
                self.hidden_size if self.representation_size is None else \
                    self.representation_size,
                num_classes
            )
        else:
            mlp_head = nn.Identity()
        super(VisionTransformer, self).__init__(OrderedDict([
            ('resnet', resnet), ('linear_projector', linear_projector),
            # ('patching', patching),
            # ('embedding_layer', embedding),
            ('encoder', encoder), ('fetch_class', fetch_class),
            ('pre_logits', pre_logits), ('mlp_head', mlp_head)
        ]), *args, **kwargs
        )

    def find_a_resnet(self, in_channels):
        """根据参数指定来分配ResNet块
        :return ResNet网络，输出通道数
        """
        if self.resnet_type == 'resnet50':
            # resnet_width = int(64 * resnet_width)
            # resnet = torchvision.models.resnet50(
            #     weights=torchvision.models.ResNet50_Weights.DEFAULT
            # )
            # resnet = nn.Sequential(
            #     nn.Conv2d(input_channel, resnet_width, 7, 2, bias=False),
            #     nn.GroupNorm(64, resnet_width),
            #     nn.ReLU(),
            #     nn.MaxPool2d(3, 2, )
            # )
            resnet = ResNet50(in_channels, [3, 4, 9])
            del resnet.avgpool, resnet.flatten, resnet.output
            output_channels = resnet.in_channels
        else:
            resnet = nn.Identity()
            output_channels = in_channels
        return resnet, output_channels

    def __get_model_config(self, which, input_size):
        assert which in supported_models, f'不支持的模型类型{which}，支持的模型类型包括{supported_models}'
        if which == 'r50_b16':
            self.__get_r50_b16_config(input_size)
        elif which == 'b16':
            self.__get_b16_config(input_size)
        else:
            pass

    def __get_b16_config(self, input_size):
        """指定ViT_B16的配置"""
        self.resnet_type = None
        self.patches_size = 16
        self.hidden_size = 768
        self.tran_emb_size = (input_size[1] // self.patches_size) * (input_size[2] // self.patches_size)
        self.tran_mlp_dim = 3072
        self.tran_num_heads = 12
        self.tran_num_layers = 12
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.0
        self.classifier = 'token'
        self.representation_size = None

    def __get_r50_b16_config(self, input_size):
        """Returns the Resnet50 + ViT-B/16 configuration."""
        self.__get_b16_config(input_size)
        self.resnet_type = 'resnet50'
        self.tran_attention_dropout_rate = 0.1
        self.patches_size = 1
        self.resnet_num_layers = (3, 4, 9)
        self.resnet_width_factor = 1

    # def forward(self, inputs):
    #     # (Possibly partial) ResNet root.
    #     # if self.resnet is not None:
    #     #     width = int(64 * self.resnet.width_factor)
    #     #
    #     #     # Root block.
    #     #     inputs = models_resnet.StdConv(
    #     #         features=width,
    #     #         kernel_size=(7, 7),
    #     #         strides=(2, 2),
    #     #         use_bias=False,
    #     #         name='conv_root')(
    #     #         inputs)
    #     #     inputs = nn.GroupNorm(name='gn_root')(inputs)
    #     #     inputs = nn.relu(inputs)
    #     #     inputs = nn.max_pool(inputs, window_shape=(3, 3), strides=(2, 2), padding='SAME')
    #     #
    #     #     # ResNet stages.
    #     #     if self.resnet.num_layers:
    #     #         inputs = models_resnet.ResNetStage(
    #     #             block_size=self.resnet.num_layers[0],
    #     #             nout=width,
    #     #             first_stride=(1, 1),
    #     #             name='block1')(
    #     #             inputs)
    #     #         for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
    #     #             inputs = models_resnet.ResNetStage(
    #     #                 block_size=block_size,
    #     #                 nout=width * 2 ** i,
    #     #                 first_stride=(2, 2),
    #     #                 name=f'block{i + 1}')(
    #     #                 inputs)
    #
    #     # We can merge s2d+emb into a single conv; it's the same.
    #     inputs = self.linear_projector(inputs)
    #
    #     # Here, inputs is a grid of embeddings.
    #
    #     # (Possibly partial) Transformer.
    #     if self.encoder is not None:
    #         bs, c, h, w = inputs.shape
    #         # 将序列长度移动到第1维，通道维移动到第2维
    #         inputs = inputs.reshape([bs, h * w, c])
    #
    #         # If we want to add a class token, add it here.
    #         if self.classifier in ['token', 'token_unpooled']:
    #             # cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
    #             cls = torch.zeros(1, 1, c).repeat(bs, 1, 1)
    #             # cls = self.ini.tile(cls, [bs, 1, 1])
    #             inputs = torch.cat([cls, inputs], 1)
    #
    #         inputs = self.encoder(inputs)
    #
    #     if self.classifier == 'token':
    #         # 0号位即为分类标签
    #         inputs = inputs[:, 0]
    #     elif self.classifier == 'gap':
    #         inputs = torch.mean(inputs, dim=list(range(1, len(inputs.shape) - 1)))  # (1,) or (1,2)
    #     elif self.classifier in ['unpooled', 'token_unpooled']:
    #         pass
    #     else:
    #         raise ValueError(f'Invalid classifier={self.classifier}')
    #     inputs = self.pre_logits(inputs)
    #     # if self.representation_size is not None:
    #     #     inputs = nn.Dense(features=self.representation_size, name='pre_logits')(inputs)
    #     #     inputs = nn.tanh(inputs)
    #     # else:
    #     #     inputs = IdentityLayer(name='pre_logits')(inputs)
    #     return self.mlp_head(inputs)
    #     # if self.num_classes:
    #     #     inputs = nn.Dense(
    #     #         features=self.num_classes,
    #     #         name='head',
    #     #         kernel_init=nn.initializers.zeros,
    #     #         bias_init=nn.initializers.constant(self.head_bias_init))(inputs)
    #     # return inputs

#
# inputs = torch.rand([4, 3, 224, 224])
# transformer = VisionTransformer('b16', 3, num_classes=784)
# inputs = transformer(inputs)
# print(inputs.shape)
