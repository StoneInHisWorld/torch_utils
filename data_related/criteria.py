import torch
from skimage.metrics import (structural_similarity as ssim,
                             peak_signal_noise_ratio as psnr)


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
    y = Y.cpu().numpy().squeeze()
    y_hat = Y_HAT.cpu().numpy().squeeze()
    result = torch.tensor([psnr(i, j, data_range=1) for i, j in zip(y, y_hat)])
    if size_averaged:
        return torch.sum(result)
    else:
        return result
