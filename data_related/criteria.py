import functools
import warnings

import numpy as np
import torch
from skimage.metrics import (structural_similarity as ssim,
                             peak_signal_noise_ratio as psnr)
from scipy.stats import pearsonr
from scipy.stats import ConstantInputWarning

from data_related.ds_operation import normalize
from utils.func.pytools import warning_handler

"""以下评价指标会将torch.Tensor转换成numpy.Array，因此使用此处的评价指标无法求导。
若有此需求，请转至networks.layers层寻找相关功能"""


def ARGMAX(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged=True) -> float:
    y_hat = torch.argmax(Y_HAT, dim=1)
    y = torch.argmax(Y, dim=1)
    cmp = (y_hat == y).type(Y.dtype)
    if size_averaged:
        return float(sum(cmp))
    else:
        return cmp


def SSIM(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged: bool = True):
    y = Y.cpu().numpy().squeeze()
    y_hat = Y_HAT.cpu().numpy().squeeze()
    if len(y.shape) == 2 and len(y_hat.shape) == 2:
        # 如果只有一张图片
        result = torch.tensor(ssim(y, y_hat, channel_axis=0, data_range=1))
    else:
        result = torch.tensor([ssim(i, j, channel_axis=0, data_range=1) for i, j in zip(y, y_hat)])
    if size_averaged:
        return torch.sum(result)
    else:
        return result


def PSNR(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged: bool = True):
    y = normalize(Y).cpu().numpy()
    y_hat = normalize(Y_HAT).cpu().numpy()
    if len(y.shape) == 2 and len(y_hat.shape) == 2:
        # 如果只有一张图片
        result = torch.tensor(psnr(y, y_hat, data_range=1))
    else:
        result = torch.tensor([psnr(i, j, data_range=1) for i, j in zip(y, y_hat)])
    if size_averaged:
        return torch.sum(result)
    else:
        return result


def PCC(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged: bool = True):
    y = normalize(Y).cpu().numpy()
    y_hat = normalize(Y_HAT).cpu().numpy()
    with warnings.catch_warnings(record=True) as warning_filter:
        warnings.simplefilter("default", ConstantInputWarning)
        # if len(y.shape) == 2 and len(y_hat.shape) == 2:
        #     # 如果只有一张图片
        #     result = torch.tensor(pearsonr(y.flatten(), y_hat.flatten())[0])
        # else:
        #     result = torch.tensor([pearsonr(i.flatten(), j.flatten())[0] for i, j in zip(y, y_hat)])
        #     if warning_filter:
        #         for warning in warning_filter:
        #             if issubclass(warning.category, ConstantInputWarning):
        #                 print(f'计算皮尔逊相关系数时遇到常数输入，此时相关系数无定义！输入如下：\n'
        #                       f'Y_HAT:{Y_HAT.item()}\nY:{Y.item()}')

        def msg_printer(*input_args):
            constant_msg = ''
            y, y_hat = input_args
            constant_msg += f'Y:{y[0]}' if np.all(y == y[0]) else ''
            constant_msg += f'Y_HAT:{y_hat[0]}' if np.all(y_hat == y_hat[0]) else ''
            print(f'\n计算皮尔逊相关系数时遇到常数输入，此时相关系数无定义！常熟输入如下：\n{constant_msg}')

        pearsonr_impl = functools.partial(
            warning_handler, func=lambda y, y_hat: pearsonr(y, y_hat)[0],
            category=ConstantInputWarning, warning_filter=warning_filter, warning_msg_printer=msg_printer
        )
        if len(y.shape) == 2 and len(y_hat.shape) == 2:
            # 如果只有一张图片
            result = torch.tensor(pearsonr_impl(y.flatten(), y_hat.flatten()))
        else:
            result = torch.tensor([pearsonr_impl(i.flatten(), j.flatten()) for i, j in zip(y, y_hat)])
    if size_averaged:
        return torch.sum(result)
    else:
        return result
