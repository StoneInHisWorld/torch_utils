import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class KVCache_AttentionHead(nn.Module):

    def __init__(self,
                 embed_size, head_size,
                 bias: bool = False, dropout=0.1, auto_regressive: bool = True,
                 device=None, dtype=None):
        """对键值对进行缓存以加速自回归预测的注意力头
        来源：https://mp.weixin.qq.com/s/oO7qscdi-StPP1qqhidKwA?scene=25&sessionid=#wechat_redirect
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias, **factory_kwargs)
        self.query = nn.Linear(embed_size, head_size, bias, **factory_kwargs)
        self.value = nn.Linear(embed_size, head_size, bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.auto_reg = auto_regressive

    def forward(self, queries, keys, values):
        """
        自注意力头前向传播
        """
        if torch.is_grad_enabled():
            return self._grad_enabled_forward(queries, keys, values)
        else:
            return self._grad_disabled_forward(queries, keys, values)

    def _grad_disabled_forward(self, q, k, v):
        raise NotImplementedError("注意力头尚未实现预测！")

    def _grad_enabled_forward(self, q, k, v):
        assert hasattr(self, 'tril'), "在进行这一批次训练之前，请先设置注意力掩码形状！"
        _, ql, _ = q.shape
        _, kl, _ = k.shape
        wei = q @ k.transpose(2, 1) / self.head_size ** 0.5
        wei.masked_fill_(self.tril[:ql, :kl] == 0, float('-inf'))
        wei = F.softmax(wei, dim=2)
        # (B, cache_size, cache_size)
        wei = self.dropout(wei)
        return wei @ v

    def refresh_cache(self, batch_size, k_cache_size, v_cache_size, device):
        # 根据缓存大小初始化缓存
        self.k_cache = torch.zeros(batch_size, k_cache_size, self.head_size, device=device)
        self.v_cache = torch.zeros(batch_size, v_cache_size, self.head_size, device=device)
        self.cache_index = 0

    def set_mask(self, ql, kl, device):
        # QK^{T}矩阵掩膜
        self.register_buffer('tril', torch.tril(torch.ones(ql, kl, device=device)))


class KVStatic_KVCAH(KVCache_AttentionHead):
    """
    对键值对进行缓存以加速自回归预测的多头注意力层
    此层认为，在自回归预测时，直到调用refresh_cache()，接收的键值对都是不变的，问询是可变的，
    来源：https://mp.weixin.qq.com/s/oO7qscdi-StPP1qqhidKwA?scene=25&sessionid=#wechat_redirect
    """

    def forward(self, queries, keys, values):
        """
        训练时，一次性传入所有查询、键值对；自回归预测时，每次传入一个问询向量
        """
        # B, T, _ = queries.shape  # B, 1 (T), num_head
        # k = self.key(keys)
        q = self.query(queries)
        # v = self.value(values)
        if self.auto_reg and not torch.is_grad_enabled():
            # 如果进行自回归预测
            # 存储键值的线性映射以节省自回归预测的计算复杂度
            if self.last_keys is not None:
                assert (self.last_keys == keys).all(), "静态KV要求每次传入的键组都是同一批次！"
            else:
                self.last_keys = keys
                self.k_cache = self.key(keys)
            if self.last_values is not None:
                assert (self.last_values == values).all(), "静态KV要求每次传入的值组都是同一批次！"
            else:
                self.last_values = values
                self.v_cache = self.value(values)
            k = self.k_cache
            v = self.v_cache
        else:
            k = self.key(keys)
            v = self.value(values)
        return super().forward(q, k, v)

    def _grad_disabled_forward(self, q, k, v):
        # assert hasattr(self, 'k_cache'), "自回归预测之前，请先刷新输出头！"
        # if self.cache_index + T <= self.cache_size:
        #     # 如果当前缓存大小，则原地更新缓存
        #     self.k_cache[:, self.cache_index: self.cache_index + T, :] = k
        #     self.v_cache[:, self.cache_index: self.cache_index + T, :] = v
        # else:
        #     # 将token向后移动一步
        #     shift = self.cache_index + T - self.cache_size
        #     # 因为输入的序列的长度始终为1，因此Shift将始终为1
        #     # 训练时输入的序列长度可不为1
        #     self.k_cache[:, :-shift, :] = self.k_cache[:, shift:, :].clone()
        #     self.v_cache[:, :-shift, :] = self.v_cache[:, shift:, :].clone()
        #     self.k_cache[:, -T:, :] = k
        #     self.v_cache[:, -T:, :] = v
        # # 更新缓存索引
        # self.cache_index = min(self.cache_index + T, self.cache_size)
        # attn_weights = q @ self.k_cache.transpose(2, 1) / self.head_size ** 0.5
        # attn_weights.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
        # attn_weights = F.softmax(attn_weights, dim=2)
        # # (B, cache_size, cache_size)
        # attn_weights = self.dropout(attn_weights)
        # return attn_weights @ self.v_cache
        assert hasattr(self, 'cache_index'), "自回归预测之前，请先刷新输出头！"
        assert q.shape[1] == 1, ("自回归预测的每个样本要求为序列长度为1！"
                                 "在自回归预测中，逐步输入每条问询向量再拼合以得到整个注意力输出。")
        # 更新缓存索引
        self.cache_index += 1
        k = self.k_cache[:, :self.cache_index]
        v = self.v_cache[:, :self.cache_index]
        # attn_weights = q.unsqueeze(1) @ k.transpose(2, 1) / self.head_size ** 0.5
        attn_weights = q @ k.transpose(2, 1) / self.head_size ** 0.5
        # attn_weights.masked_fill_(
        #     self.tril[:self.cache_index, :self.cache_index] == 0,
        #     float('-inf')
        # )
        attn_weights = F.softmax(attn_weights, dim=2)
        # (B, cache_size, cache_size)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ v

    # def _grad_enabled_forward(self, q, k, v):
    #     assert hasattr(self, 'tril'), "在进行这一批次训练之前，请先设置注意力掩码形状！"
    #     _, ql, _ = q.shape
    #     _, kl, _ = k.shape
    #     wei = q @ k.transpose(2, 1) / self.head_size ** 0.5
    #     wei.masked_fill_(self.tril[:ql, :kl] == 0, float('-inf'))
    #     wei = F.softmax(wei, dim=2)
    #     # (B, cache_size, cache_size)
    #     wei = self.dropout(wei)
    #     return wei @ v

    def refresh_cache(self, batch_size, k_cache_size, v_cache_size, device):
        # if self.k_cache is None or self.v_cache is None:
        # 使用缓存大小初始化缓存
        # self.k_cache = None
        # self.v_cache = None
        self.last_keys = None
        self.last_values = None
        self.cache_index = 0

    # def set_mask(self, ql, kl):
    #     # QK^{T}矩阵掩膜
    #     self.register_buffer('tril', torch.tril(torch.ones(ql, kl)))


class SelfAttention_KVCAH(KVCache_AttentionHead):
    """对键值对进行缓存以加速自回归运算的多头自注意力层
    来源：https://mp.weixin.qq.com/s/oO7qscdi-StPP1qqhidKwA?scene=25&sessionid=#wechat_redirect
    """

    #
    # def __init__(self,
    #              embed_size, head_size,
    #              bias: bool = False, dropout=0.1,
    #              device=None, dtype=None):
    #     """对键值对进行缓存以加速自回归运算的多头注意力层
    #     来源：https://mp.weixin.qq.com/s/oO7qscdi-StPP1qqhidKwA?scene=25&sessionid=#wechat_redirect
    #     """
    #     factory_kwargs = {'device': device, 'dtype': dtype}
    #     super().__init__(
    #         embed_size, head_size, bias, dropout,
    #              device, dtype)
    #     self.head_size = head_size
    #     # self.cache_size = cache_size
    #     self.key = nn.Linear(embed_size, head_size, bias, **factory_kwargs)
    #     self.query = nn.Linear(embed_size, head_size, bias, **factory_kwargs)
    #     self.value = nn.Linear(embed_size, head_size, bias, **factory_kwargs)
    #     self.dropout = nn.Dropout(dropout)
    #     # self.k_cache = None
    #     # self.v_cache = None
    #     # self.cache_index = 0

    # def forward(self, x):
    #     B, T, C = x.shape  # 形状: B, 1, C
    #     k = self.key(x)
    #     q = self.query(x)
    #     v = self.value(x)
    #     # 如果缓存为空则初始化
    #     if self.k_cache is None or self.v_cache is None:
    #         # 使用固定大小初始化缓存
    #         self.k_cache = torch.zeros(B, block_size, self.head_size, device=x.device)
    #         self.v_cache = torch.zeros(B, block_size, self.head_size, device=x.device)
    #         self.cache_index = 0
    #     return out

    def forward(self, queries, keys, values):
        """
        自注意力头前向传播
        """
        assert (queries == keys).all() and (values == keys).all(), "自注意力前向传播要求问询、键与值都相同！"
        # B, T, _ = queries.shape  # B, 1 (T), num_head
        k = self.key(keys)
        q = self.query(queries)
        v = self.value(values)
        # if self.k_cache is None or self.v_cache is None:
        #     # 使用缓存大小初始化缓存
        #     self.k_cache = torch.zeros(B, self.cache_size, self.head_size, device=q.device)
        #     self.v_cache = torch.zeros(B, self.cache_size, self.head_size, device=q.device)
        #     self.cache_index = 0
        # if self.cache_index + T <= self.cache_size:
        #     # 如果当前缓存大小，则原地更新缓存
        #     self.k_cache[:, self.cache_index: self.cache_index + T, :] = k
        #     self.v_cache[:, self.cache_index: self.cache_index + T, :] = v
        # else:
        #     # 将token向后移动一步
        #     shift = self.cache_index + T - self.cache_size
        #     # 因为输入的序列的长度始终为1，因此Shift将始终为1
        #     # 训练时输入的序列长度可不为1
        #     self.k_cache[:, :-shift, :] = self.k_cache[:, shift:, :].clone()
        #     self.v_cache[:, :-shift, :] = self.v_cache[:, shift:, :].clone()
        #     self.k_cache[:, -T:, :] = k
        #     self.v_cache[:, -T:, :] = v
        # # 更新缓存索引
        # self.cache_index = min(self.cache_index + T, self.cache_size)
        # wei = q @ self.k_cache.transpose(2, 1) / self.head_size ** 0.5
        # wei.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
        # wei = F.softmax(wei, dim=2)
        # # (B, block_size, block_size)
        # wei = self.dropout(wei)
        # return wei @ self.v_cache
        if self.auto_reg and not torch.is_grad_enabled():
            return self._grad_disabled_forward(q, k, v)
        else:
            return super()._grad_enabled_forward(q, k, v)
        # return super().forward(q, k, v)

    def _grad_disabled_forward(self, q, k, v):
        assert hasattr(self, 'k_cache'), "自回归预测之前，请先刷新输出头！"
        # assert len(q.shape) == 2, ("自回归预测的每个样本要求为二维向量！"
        #                            "在自回归预测中，逐步输入每条问询向量再拼合以得到整个注意力输出。")
        assert q.shape[1] == 1, ("自回归预测的每个样本要求为序列长度为1！"
                                 "在自回归预测中，逐步输入每条问询向量再拼合以得到整个注意力输出。")
        # if self.cache_index + T <= self.cache_size:
        #     # 如果当前缓存大小，则原地更新缓存
        #     self.k_cache[:, self.cache_index: self.cache_index + T, :] = k
        #     self.v_cache[:, self.cache_index: self.cache_index + T, :] = v
        # else:
        #     # 将token向后移动一步
        #     shift = self.cache_index + T - self.cache_size
        #     # 因为输入的序列的长度始终为1，因此Shift将始终为1
        #     # 训练时输入的序列长度可不为1
        #     self.k_cache[:, :-shift, :] = self.k_cache[:, shift:, :].clone()
        #     self.v_cache[:, :-shift, :] = self.v_cache[:, shift:, :].clone()
        #     self.k_cache[:, -T:, :] = k
        #     self.v_cache[:, -T:, :] = v
        # # 原地更新缓存
        # self.k_cache[:, self.cache_index: self.cache_index + T, :] = k
        # self.v_cache[:, self.cache_index: self.cache_index + T, :] = v
        # # 更新缓存索引
        # self.cache_index = min(self.cache_index + T, self.cache_size)
        # attn_weights = q @ self.k_cache.transpose(2, 1) / self.head_size ** 0.5
        # attn_weights.masked_fill_(self.tril[:T, :T] == 0, float('-inf'))
        # attn_weights = F.softmax(attn_weights, dim=2)
        # # (B, cache_size, cache_size)
        # attn_weights = self.dropout(attn_weights)
        # return attn_weights @ self.v_cache
        # 原地更新缓存
        self.k_cache[:, self.cache_index, :] = k.squeeze()
        self.v_cache[:, self.cache_index, :] = v.squeeze()
        # 更新缓存索引
        self.cache_index += 1
        k = self.k_cache[:, :self.cache_index]
        v = self.v_cache[:, :self.cache_index]
        # attn_weights = q.unsqueeze(1) @ k.transpose(2, 1) / self.head_size ** 0.5
        attn_weights = q @ k.transpose(2, 1) / self.head_size ** 0.5
        # attn_weights.masked_fill_(self.tril[:self.cache_index, :self.cache_index] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=2)
        # (B, cache_size, cache_size)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ v

    # def _grad_enabled_forward(self, q, k, v):
    #     assert hasattr(self, 'tril'), "在进行这一批次训练之前，请先设置注意力掩码形状！"
    #     _, ql, _ = q.shape
    #     # _, kl, _ = k.shape
    #     wei = q @ k.transpose(2, 1) / self.head_size ** 0.5
    #     wei.masked_fill_(self.tril[:ql, :ql] == 0, float('-inf'))
    #     wei = F.softmax(wei, dim=2)
    #     # (B, cache_size, cache_size)
    #     wei = self.dropout(wei)
    #     return wei @ v

    # def refresh_cache(self, batch_size, k_cache_size, v_cache_size, device):
    #     # 根据缓存大小初始化缓存
    #     # +1是为了预留空间给最后一批kv值
    #     self.k_cache = torch.zeros(batch_size, k_cache_size + 1, self.head_size, device=device)
    #     self.v_cache = torch.zeros(batch_size, v_cache_size + 1, self.head_size, device=device)
    #     self.cache_index = 0

    def set_mask(self, ql, kl, device):
        # QK^{T}矩阵掩膜
        assert ql == kl, "自注意力掩码需为方阵！"
        super().set_mask(ql, kl, device=device)
        # self.register_buffer('tril', torch.tril(torch.ones(ql, kl)))


# 多头注意力
class KVCacheMultiHeadAttention(nn.Module):

    def __init__(
            self,
            num_head, embed_size, self_attn: bool, bias: bool = False,
            dropout: float = 0.1, auto_regressive: bool = True,
            device=None, dtype=None
    ):
        """使用KV缓存头的多头注意力"""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert embed_size % num_head == 0, f"模型词语的维度{embed_size}应该能整除头数目{num_head}！"
        head_size = embed_size // num_head

        self.num_heads = num_head
        self.batch_first = True
        self._qkv_same_embed_dim = True

        self.sa_head = nn.ModuleList([
            SelfAttention_KVCAH(
                embed_size, head_size, bias, dropout, auto_regressive, **factory_kwargs)
            if self_attn else KVStatic_KVCAH(
                embed_size, head_size, bias, dropout, auto_regressive, **factory_kwargs
            )
            for _ in range(num_head)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(
            embed_size, embed_size,
            # bias,
            **factory_kwargs
        )
        # self.in_proj_bias = self.out_proj.bias
        self.in_proj_weight = Parameter(torch.empty(3 * embed_size, **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_size, **factory_kwargs))

    def forward(self, q, k, v):
        out = torch.cat([head(q, k, v) for head in self.sa_head], dim=-1)
        out = self.dropout(self.out_proj(out))
        return out

    def refresh_head(self, batch_size, ql, kl, vl, device):
        # if self.k_cache is None or self.v_cache is None:
        # 使用缓存大小初始化缓存
        for head in self.sa_head:
            if not torch.is_grad_enabled():
                head.refresh_cache(batch_size, kl, vl, device)
            else:
                head.set_mask(ql, kl, device)

#
# embed_size = 128
# cache = 10
# src_len = 20
# tgt_len = 10
# auto_reg_len = 30
# batch_size = 32
# self_attn_model = KVCacheMultiHeadAttention(
#     8, embed_size, True, True
# )
# kvstatic_model = KVCacheMultiHeadAttention(
#     8, embed_size, False, True
# )
# src = torch.randn(batch_size, src_len, embed_size)
# tgt = torch.randn(batch_size, tgt_len, embed_size)
#
# with torch.enable_grad():
#     self_attn_model.refresh_head(
#         batch_size, src_len, src_len, src_len, src.device
#     )
#     kvstatic_model.refresh_head(
#         batch_size, tgt_len, src_len, src_len, src.device
#     )
#     self_attn_out = self_attn_model(src, src, src)
#     kv_static_out = kvstatic_model(tgt, src, src)
#
# outputs = []
# with torch.no_grad():
#     for i in range(tgt_len):
#         t = tgt[:, i]
#         inputs = self_attn_model(t, t, t)
#         outputs.append(inputs)
# outputs = torch.cat(outputs, dim=1)
# print(outputs.shape)
#
# outputs = []
# with torch.no_grad():
#     for i in range(tgt_len):
#         t = tgt[:, i]
#         inputs = kvstatic_model(t, src, src)
#         outputs.append(inputs)
# outputs = torch.cat(outputs, dim=1)
# print(outputs.shape)
