import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

import utils.func.torch_tools as ttools
from layers.kvcache_MultiheadAttention import KVCacheMultiHeadAttention


class KVCache_TransformerEncoderLayer(TransformerEncoderLayer):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = False, auto_regressive: bool = True,
                 device=None, dtype=None) -> None:
        """由torch.nn.transformer.TransformerEncoderLayer改编而来"""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            d_model, nhead, dim_feedforward, dropout, layer_norm_eps=layer_norm_eps, batch_first=True,
            norm_first=norm_first, bias=bias, **factory_kwargs
        )
        del self.self_attn
        self.batch_first = True
        self.self_attn = KVCacheMultiHeadAttention(
            nhead, d_model, True, bias,dropout, auto_regressive, **factory_kwargs
        )
        # # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        #
        # self.norm_first = norm_first
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation_relu_or_gelu = 1
        elif activation == 'gelu':
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = ttools.get_activation(activation)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        if attn_mask is not None:
            warnings.warn('KV缓存注意力机制中自动生成注意力掩码，不需要赋值！')
        if key_padding_mask is not None:
            warnings.warn('KV缓存注意力机制中没有设置键填充掩码！')
        if is_causal:
            warnings.warn('KV缓存注意力机制中没有causal机制！')
        # bs, l, _ = x.shape
        # self.self_attn.refresh_head(bs, l, l, x.device)
        x = self.self_attn(x, x, x)
        # x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)


class KVCache_TransformerDecoderLayer(TransformerDecoderLayer):

    def __init__(self,
                 d_model: int, nhead: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu',
                 layer_norm_eps: float = 1e-5, norm_first: bool = False, bias: bool = True,
                 auto_regressive: bool = True, device=None, dtype=None) -> None:
        """由torch.nn.transformer.TransformerDecoderLayer改编而来"""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            d_model, nhead, dim_feedforward, dropout, layer_norm_eps=layer_norm_eps, batch_first=True,
            norm_first=norm_first, bias=bias, **factory_kwargs
        )
        del self.self_attn, self.multihead_attn
        self.batch_first = True
        self.self_attn = KVCacheMultiHeadAttention(
            nhead, d_model, True, bias, dropout, auto_regressive, **factory_kwargs
        )
        self.multihead_attn = KVCacheMultiHeadAttention(
            nhead, d_model, False, bias, dropout, auto_regressive, **factory_kwargs
        )
        self.activation = ttools.get_activation(activation)

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        if attn_mask is not None:
            warnings.warn('KV缓存注意力机制中自动生成注意力掩码，不需要赋值！')
        if key_padding_mask is not None:
            warnings.warn('KV缓存注意力机制中没有设置键填充掩码！')
        if is_causal:
            warnings.warn('KV缓存注意力机制中没有causal机制！')
        x = self.self_attn(x, x, x)
        # x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        if attn_mask is not None:
            warnings.warn('KV缓存注意力机制中自动生成注意力掩码，不需要赋值！')
        if key_padding_mask is not None:
            warnings.warn('KV缓存注意力机制中没有设置键填充掩码！')
        if is_causal:
            warnings.warn('KV缓存注意力机制中没有causal机制！')
        x = self.multihead_attn(x, mem, mem)
        # x = self.multihead_attn(x, mem, mem)[0]
        return self.dropout2(x)


class KVCacheTransformerBlock(nn.Transformer):
    """
    对键值对进行缓存以加速自回归预测的Transformer块
    来源：https://mp.weixin.qq.com/s/oO7qscdi-StPP1qqhidKwA?scene=25&sessionid=#wechat_redirect
    """

    def __init__(
        self,
        src_len, tgt_len,
        d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
        num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
        activation: str = 'relu', layer_norm_eps: float = 1e-5, auto_regressive: bool = True,
        norm_first: bool = False, bias: bool = False, device=None, dtype=None
    ):
        """
        对键值对进行缓存以加速自回归预测的Transformer块
        来源：https://mp.weixin.qq.com/s/oO7qscdi-StPP1qqhidKwA?scene=25&sessionid=#wechat_redirect
        改编自：https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer

        Args:
            src_len: 输入的序列长
            tgt_len: 输出的序列长
            d_model: 词向量维度.
            nhead: 多头注意力的分头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前向反馈网络的维度
            dropout: 暂退法所用比例
            activation: 模型使用的激活函数
            layer_norm_eps: 层正则化所使用的数值稳定项
            auto_regressive: 是否将此模型应用于自回归预测
            norm_first: 当设置为True时，编码器和解码器在进行注意力分数计算和前向反馈计算前，会先进行层正则化
            bias: 如果设置为False，则Linear和LayerNorm层不会学习额外的偏置项
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        encoder = nn.TransformerEncoder(
            KVCache_TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                norm_first, bias, False, **factory_kwargs
            ), num_encoder_layers, nn.LayerNorm(d_model, layer_norm_eps), False
        )
        decoder = nn.TransformerDecoder(
            KVCache_TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
                norm_first, bias, auto_regressive, **factory_kwargs
            ), num_decoder_layers, nn.LayerNorm(d_model, layer_norm_eps)
        )
        self.src_len = src_len
        self.tgt_len = tgt_len
        super().__init__(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
            dropout, activation, encoder, decoder, layer_norm_eps, True,
            norm_first, bias, **factory_kwargs
        )

    def refresh_head(self, batch_size, device):
        # if self.k_cache is None or self.v_cache is None:
        sl, tl = self.src_len, self.tgt_len
        # 使用缓存大小初始化缓存
        for layer in self.encoder.layers:
            layer.self_attn.refresh_head(
                batch_size, sl, sl, sl, device
            )
        for layer in self.decoder.layers:
            layer.self_attn.refresh_head(
                batch_size, tl, tl, tl, device
            )
            layer.multihead_attn.refresh_head(
                batch_size, tl, sl, sl, device
            )


def auto_regression(transformer, src, init_q, stop_fn, keep_init_q: bool = False):
    """
    自回归算法

    Args:
        transformer: 含有编码器和解码器的transformer块。
            使用编码器形成记忆，之后解码器根据`init_q`和记忆形成输出
        src: 模型输入
        init_q: 解码器形成输出的初始询问向量
        stop_fn: 停止函数，需为返回布尔值的可调用对象。每次解码器形成一条输出向量后会调用本函数判断输出是否停止。
            签名要求为：def stop_fn(output: Tensor, progress: int) -> bool
            output为解码器最新一条输出向量；progress为编码器输出进度，其值等于目前生成的输出向量个数。
        keep_init_q: 是否要保留初始询问向量
    """
    mem = transformer.encoder(src)
    outputs = [init_q]
    progress = 0
    while not stop_fn(outputs[-1], progress):
        outputs.append(transformer.decoder(outputs[-1], mem))
        progress += 1
    if keep_init_q:
        return torch.cat(outputs, dim=1)
    else:
        return torch.cat(outputs, dim=1)[:, 1:]


# batch_size = 32
# n_elayer = 1
# n_dlayer = 1
# nhead = 4
# d_model = 128
# src_len = 10
# tgt_len = 20
# kv_encoder = nn.TransformerEncoder(KVCache_TransformerEncoderLayer(
#         d_model, nhead
# ), n_elayer, nn.LayerNorm([src_len, d_model]),
#     enable_nested_tensor=False
# )
# kv_decoder = nn.TransformerDecoder(KVCache_TransformerDecoderLayer(
#         d_model, nhead
# ), n_dlayer, nn.LayerNorm([tgt_len, d_model])
# )
# kv_transformer = KVCacheTransformerBlock(
#     src_len, tgt_len, d_model, nhead, n_elayer, n_dlayer, 4096, 0.5, 'gelu'
# )
#
# # # 官方实现的Transformer
# # encoder = nn.TransformerEncoder(
# #     TransformerEncoderLayer(d_model, nhead, batch_first=True), n_layers,
# #     nn.LayerNorm([seq_len, d_model]), enable_nested_tensor=False
# # )
# # decoder = nn.TransformerDecoder(
# #     TransformerDecoderLayer(d_model, nhead, batch_first=True), n_layers,
# #     nn.LayerNorm([tgt_seq_len, d_model])
# # )
# src = torch.randn(batch_size, src_len, d_model)
# tgt = torch.randn(batch_size, tgt_len, d_model)
#
# # # 训练测试
# # with torch.enable_grad():
# #     # 使用官方实现的Transformer作为范例
# #     mem = encoder(src)
# #     out = decoder(tgt, mem)
# #
# #     # 初始化掩码
# #     for layer in kv_encoder.layers:
# #         layer.self_attn.refresh_head(batch_size, seq_len, seq_len, src.device)
# #     for layer in kv_decoder.layers:
# #         layer.self_attn.refresh_head(batch_size, tgt_seq_len, tgt_seq_len, src.device)
# #         layer.multihead_attn.refresh_head(batch_size, tgt_seq_len, seq_len, src.device)
# #     kv_mem = kv_encoder(src)
# #     kv_out = kv_decoder(tgt, kv_mem)
#
# # 测试层
# # with torch.enable_grad():
# #     for layer in kv_encoder.layers:
# #         layer.self_attn.refresh_head(
# #             batch_size, src_len, src_len, src_len, src.device
# #         )
# #     for layer in kv_decoder.layers:
# #         layer.self_attn.refresh_head(
# #             batch_size, tgt_len, tgt_len, tgt_len, src.device
# #         )
# #         layer.multihead_attn.refresh_head(
# #             batch_size, tgt_len, src_len, src_len, src.device
# #         )
# #
# #     kv_mem = kv_encoder(src)
# #     kv_output = kv_decoder(tgt, kv_mem)
# #     print(kv_output.shape)
# #
# # with torch.enable_grad():
# #     kv_transformer.refresh_head(batch_size, src.device)
# #     kv_output = kv_transformer(src, tgt)
# #     print(kv_output.shape)
#
# with torch.no_grad():
#     kv_transformer.refresh_head(batch_size, src.device)
#     kv_output = auto_regression(
#         kv_transformer, src, torch.zeros(batch_size, 1, d_model),
#         lambda last_o, progress: progress >= tgt_len, False
#     )
#     print(kv_output.shape)
#
# # with torch.no_grad():
# #     # # 使用官方实现的Transformer作为范例
# #     # mem = encoder(src)
# #     # out = decoder(tgt, mem)
# #
# #     # 初始化掩码和缓存
# #     for layer in kv_encoder.layers:
# #         layer.self_attn.refresh_head(batch_size, seq_len, seq_len, src.device)
# #     # for layer in kv_decoder.layers:
# #         # layer.self_attn.refresh_head(batch_size, tgt_seq_len, tgt_seq_len, src.device)
# #         # layer.multihead_attn.refresh_head(batch_size, tgt_seq_len, seq_len, src.device)
# #
# #     outputs = []
# #     sos = torch.randn(batch_size, 1, d_model)
# #     kv_mem = kv_encoder(src)
# #     for _ in range(64):
# #         print(_)
# #         outputs.append(kv_decoder(sos, kv_mem))
# #     outputs = torch.cat(outputs)
# # kv_en_outputs = kv_encoder(data)
# # kv_de_outputs = []
# # de_outputs = []
# # for _ in range(64):
# #     print(_)
# #     kv_de_outputs.append(kv_decoder(tgt, kv_en_outputs))
# #     # de_outputs.append(decoder(tgt, en_outputs))
# # de_outputs = torch.cat(de_outputs, dim=1)
# # output = encoder(data)



