import functools
from collections import OrderedDict
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from networks import BasicNN
from layers.add_positionEmbeddings import AddPositionEmbs
from layers import KVCacheTransformerBlock, auto_regression


def SEQ_ENTROLOSS(pred, y, unwrapped_entroloss):
    pred = pred.permute(0, 2, 1)
    return unwrapped_entroloss(pred, y)


def SEQ_MSE(pred, y, unwrapped_mseloss, pixel_basis):
    pred = (pred @ pixel_basis.to(pred.device)).squeeze()
    y = y.to(torch.float32)
    return unwrapped_mseloss(pred, y)


def SEQ_L1(pred, y, unwrapped_mseloss, pixel_basis):
    pred = (pred @ pixel_basis.to(pred.device)).squeeze()
    y = y.to(torch.float32)
    return unwrapped_mseloss(pred, y)


class ITransformer(BasicNN):

    def __init__(
            self,
            in_len, in_pixel_range, out_len, out_pixel_range,
            auto_regression: bool = True, pos_emb='original', tran_kwargs=None,
            **kwargs
    ):
        if tran_kwargs is None:
            tran_kwargs = {}
        self.pixel_range = max(in_pixel_range, out_pixel_range)
        # 这里认为输入和输出图片所在像素域不重叠，因此特征集和标签集嵌入参数不共享
        d_model = tran_kwargs.pop('d_model', 512)
        f_embedding = nn.Embedding(in_pixel_range, d_model)
        l_embedding = nn.Embedding(out_pixel_range, d_model)
        if pos_emb:
            pos_embedding = AddPositionEmbs(
                pos_emb, (max(in_len, out_len), d_model)
            )
        else:
            pos_embedding = nn.Identity()
        transformer = KVCacheTransformerBlock(in_len, out_len, d_model, auto_regressive=auto_regression, **tran_kwargs)
        linear_projector = nn.Linear(d_model, out_pixel_range)
        self.pixel_basis = torch.arange(0, out_pixel_range).reshape(1, out_pixel_range, 1).to(torch.float32)
        # self.patches_size = patches_size
        self.auto_reg = auto_regression
        super().__init__(OrderedDict([
            ("f_embedding", f_embedding),
            ("l_embedding", l_embedding),
            ("pos_embedding", pos_embedding),
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
        if self.auto_reg and not torch.is_grad_enabled():
            inputs = self.pos_embedding(self.f_embedding(inputs))
            b, _, d = inputs.shape
            self.transformer.refresh_head(b, inputs.device)
            outputs = auto_regression(
                self.transformer, inputs, torch.zeros((b, 1, d), device=inputs.device),
                lambda out, progress: progress >= self.transformer.tgt_len
            )
        else:
            inputs, labels = inputs
            inputs = self.pos_embedding(self.f_embedding(inputs))
            labels = self.pos_embedding(self.l_embedding(labels))
            self.transformer.refresh_head(inputs.shape[0], inputs.device)
            outputs = self.transformer(inputs, labels)
        outputs = self.linear_projector(outputs)
        return F.softmax(outputs, 2)

    def _forward_impl(self, X, y) -> Tuple[torch.Tensor, List]:
        """前向传播实现。
        进行前向传播后，根据self._ls_fn()计算损失值，并返回。
        参考自：https://www.zhihu.com/question/584772471/answer/2900724691

        :param X: 特征集
        :param y: 标签集
        :return: （预测值， （损失值集合））
        """
        if torch.is_grad_enabled():
            pred = self((X, y))
            ls_es = [ls_fn(pred, y) for ls_fn in self.train_ls_fn_s]
        else:
            # 自回归预测
            pred = self(X)
            ls_es = [ls_fn(pred, y) for ls_fn in self.test_ls_fn_s]
        self.pixel_basis = self.pixel_basis.to(X.device)
        pred = (pred @ self.pixel_basis).squeeze()
        return pred, ls_es

    def _get_ls_fn(self, ls_args):
        train_ls_fn_s, train_ls_names, test_ls_fn_s, test_ls_names = super()._get_ls_fn(*ls_args)
        """针对训练损失函数的特殊处理"""
        try:
            # 针对交叉熵损失的特别处理
            where = train_ls_names.index('ENTRO')
            train_ls_fn_s[where] = functools.partial(
                SEQ_ENTROLOSS, unwrapped_entroloss=train_ls_fn_s[where]
            )
        except ValueError:
            pass
        try:
            # 针对均方差的特别处理
            where = train_ls_names.index('MSE')
            train_ls_fn_s[where] = functools.partial(
                SEQ_MSE, unwrapped_mseloss=train_ls_fn_s[where], pixel_basis=self.pixel_basis
            )
        except ValueError:
            pass
        try:
            # 针对均方差的特别处理
            where = train_ls_names.index('L1')
            train_ls_fn_s[where] = functools.partial(
                SEQ_L1, unwrapped_mseloss=train_ls_fn_s[where], pixel_basis=self.pixel_basis
            )
        except ValueError:
            pass
        """针对测试损失函数的特别处理"""
        try:
            where = test_ls_names.index('ENTRO')
            test_ls_fn_s[where] = functools.partial(
                SEQ_ENTROLOSS, unwrapped_entroloss=test_ls_fn_s[where]
            )
        except ValueError:
            pass
        try:
            # 针对均方差的特别处理
            where = test_ls_names.index('MSE')
            test_ls_fn_s[where] = functools.partial(
                SEQ_MSE, unwrapped_mseloss=test_ls_fn_s[where], pixel_basis=self.pixel_basis
            )
        except ValueError:
            pass
        try:
            # 针对均方差的特别处理
            where = test_ls_names.index('L1')
            test_ls_fn_s[where] = functools.partial(
                SEQ_L1, unwrapped_mseloss=test_ls_fn_s[where], pixel_basis=self.pixel_basis
            )
        except ValueError:
            pass
        return train_ls_fn_s, train_ls_names, test_ls_fn_s, test_ls_names

    def get_key_padding_mask(self, tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


# batch_size = 32
# inputs_len = 20
# labels_len = 40
# inputs_dim = 256
# labels_dim = 2
# model = ITransformer(
#     inputs_len, inputs_dim, labels_len, labels_dim, tran_kwargs={
#         "nhead": 2, "num_encoder_layers": 2, "num_decoder_layers": 2,
#         "d_model": 64, "dim_feedforward": 512
#     }
# )
# inputs = torch.randint(0, 256, [batch_size, inputs_len])
# labels = torch.randint(0, 2, [batch_size, labels_len])
# with torch.enable_grad():
#     outputs = model((inputs, labels))
#     print(outputs.shape)
# with torch.no_grad():
#     outputs = model(inputs)
#     print(outputs.shape)
