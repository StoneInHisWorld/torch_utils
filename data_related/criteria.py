import torch
from skimage.metrics import (structural_similarity as ssim,
                             peak_signal_noise_ratio as psnr)
from scipy.stats import pearsonr

import networks.layers.pcc
from data_related.data_related import normalize

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
    result = torch.tensor([ssim(i, j, channel_axis=0, data_range=1) for i, j in zip(y, y_hat)])
    if size_averaged:
        return torch.sum(result)
    else:
        return result


def PSNR(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged: bool = True):
    # y = Y.cpu().numpy().squeeze()
    # y_hat = Y_HAT.cpu().numpy().squeeze()
    y = normalize(Y).cpu().numpy()
    y_hat = normalize(Y_HAT).cpu().numpy()
    result = torch.tensor([psnr(i, j, data_range=1) for i, j in zip(y, y_hat)])
    if size_averaged:
        return torch.sum(result)
    else:
        return result


def PCC(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged: bool = True):
    y = normalize(Y).cpu().numpy()
    y_hat = normalize(Y_HAT).cpu().numpy()
    result = torch.tensor([pearsonr(i.flatten(), j.flatten())[0] for i, j in zip(y, y_hat)])
    # tensor_result = networks.layers.pcc.PCC(size_averaged)(Y_HAT, Y)
    if size_averaged:
        return torch.sum(result)
    else:
        return result
