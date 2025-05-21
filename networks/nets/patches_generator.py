import functools
from collections import OrderedDict
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from networks import BasicNN
from layers.add_positionEmbeddings import AddPositionEmbs


#
#
# class PatchTransformerDecoder(BasicNN):
#     def __init__(self, vocab_size, key_size, query_size, value_size,
#                  num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
#                  num_heads, num_layers, dropout, **kwargs):
#         self.num_hiddens = num_hiddens
#         self.num_layers = num_layers
#         self.embedding = nn.Embedding(vocab_size, num_hiddens)
#         self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
#         self.blks = nn.Sequential()
#         blk_s = [
#             DecoderBlock(key_size, query_size, value_size, num_hiddens,
#                          norm_shape, ffn_num_input, ffn_num_hiddens,
#                          num_heads, dropout, i)
#             for i in range(num_layers)
#         ]
#         self.dense = nn.Linear(num_hiddens, vocab_size)
#         super(PatchTransformerDecoder, self).__init__(**kwargs)
#
#     def init_state(self, enc_outputs, enc_valid_lens, *args):
#         return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
#
#     def forward(self, X, state):
#         X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
#         self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
#         for i, blk in enumerate(self.blks):
#             X, state = blk(X, state)
#             # 解码器自注意力权重
#             self._attention_weights[0][
#                 i] = blk.attention1.attention.attention_weights
#             # “编码器－解码器”自注意力权重
#             self._attention_weights[1][
#                 i] = blk.attention2.attention.attention_weights
#         return self.dense(X), state
#
#     @property
#     def attention_weights(self):
#         return self._attention_weights
#

def SEQ_ENTROLOSS(pred, y, wrapped_entroloss):
    pred = pred.permute(0, 2, 1)
    y = y.argmax(2)
    return wrapped_entroloss(pred, y)


class PatchGenerator(BasicNN):

    def __init__(self,
                 in_len, in_pixel_range, out_pixel_range,
                 patches_size=4, pos_emb='original', tran_kwargs=None,
                 **kwargs):
        if tran_kwargs is None:
            tran_kwargs = {}
        emb_size = max(in_pixel_range, out_pixel_range)
        if pos_emb:
            position_embedding = AddPositionEmbs(
                pos_emb, (in_len, emb_size)
            )
        else:
            position_embedding = nn.Identity()
        transformer = nn.Transformer(emb_size, batch_first=True, **tran_kwargs)
        linear_projector = nn.Linear(in_pixel_range, out_pixel_range)
        self.patches_size = patches_size
        super().__init__(OrderedDict([
            ("position_embedding", position_embedding),
            ("transformer", transformer),
            ("linear_projector", linear_projector),
        ]), **kwargs)

    def forward(self, inputs):
        """
        Args:
            inputs: （输入向量（像素类别填充至与输入向量一致），
                标签图片（像素类别填充至与输入向量一致））

        Returns:

        """
        inputs, labels = inputs
        # i_bs, i_len, i_c = inputs.shape
        # l_bs, l_len, l_c = labels.shape
        # # 将序列长度移动到第1维，通道维移动到第2维
        # inputs = inputs.permute(0, 2, 3, 1)
        # labels = labels.permute(0, 2, 3, 1)
        # inputs = inputs.reshape([i_bs, i_h * i_w, i_c])
        # labels = labels.reshape([l_bs, l_h * l_w, l_c])
        # 使用F.pad填充
        # f_kwargs = {}
        # if i_c > l_c:
        #     labels = F.pad(labels, (0, i_c - l_c, 0, 0), mode='constant', value=0)
        # elif i_c < l_c:
        #     inputs = F.pad(inputs, (0,  l_c - i_c, 0, 0), mode='constant', value=0)
        inputs = self.position_embedding(inputs)
        outputs = []
        for l_slice in torch.split(labels, self.patches_size ** 2, 1):
            outputs.append(self.transformer(inputs, l_slice))
        outputs = torch.cat(outputs, 1)
        return self.linear_projector(outputs)

    def _forward_impl(self, X, y) -> Tuple[torch.Tensor, List]:
        """前向传播实现。
        进行前向传播后，根据self._ls_fn()计算损失值，并返回。
        若要更改optimizer.zero_grad()以及backward()的顺序，请直接重载forward_backward()！
        :param X: 特征集
        :param y: 标签集
        :return: （预测值， （损失值集合））
        """
        # 填充至词维度一致
        embd_size = X.shape[2] - y.shape[2]
        if embd_size < 0:
            X = torch.nn.functional.pad(
                X, (0, -embd_size, 0, 0), mode='constant', value=0
            )
        elif embd_size > 0:
            y = torch.nn.functional.pad(
                y, (0, embd_size, 0, 0), mode='constant', value=0
            )
        # 前向传播
        if X.requires_grad:
            pred = self((X, y))
        else:
            pred = self((X, torch.zeros_like(y)))
        # 去掉标签集中填充部分以计算损失值
        if embd_size > 0:
            y = y[:, :, :-embd_size]
        return pred, [ls_fn(pred, y) for ls_fn in self.ls_fn_s]

    def _get_ls_fn(self, *args):
        ls_fn_s, loss_names, test_ls_names = super()._get_ls_fn(*args)
        try:
            where = loss_names.index('ENTRO')
            ls_fn_s[where] = functools.partial(
                SEQ_ENTROLOSS, wrapped_entroloss=ls_fn_s[where]
            )
        except ValueError:
            pass
        return ls_fn_s, loss_names, test_ls_names



#
# model = PatchGenerator(81, 256, 2)
# inputs = torch.rand([4, 81, 256])
# labels = torch.rand([4, 256, 2])
# labels = F.pad(labels, (0, 254, 0, 0), mode='constant', value=0)
# outputs = model((inputs, labels))
# print(outputs.shape)
