import torch
from torch import nn

from layers.add_positionEmbeddings import AddPositionEmbs
from networks.nets.vit import supported_classifier


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
