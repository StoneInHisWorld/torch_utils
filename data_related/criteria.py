import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


"""请勿编写无法使用CUDA加速的函数，否则会大大影响CUDA设备的训练速度！"""


def ARGMAX(Y_HAT: torch.Tensor, Y: torch.Tensor,
           size_averaged=True) -> float:
    """argmax评价标准
    计算预测张量序列和标签张量序列中，每个张量最大值的下标，并进行比较，得出相等张量的数目。

    :param Y_HAT: 预测张量序列
    :param Y: 标签张量序列
    :param size_averaged: 是否进行批量平均
    :return: 比较结果
    """
    y_hat = torch.argmax(Y_HAT, dim=1)
    y = torch.argmax(Y, dim=1)
    cmp = (y_hat == y).type(Y.dtype)
    if size_averaged:
        return float(sum(cmp))
    else:
        return cmp


def SSIM(Y_HAT: torch.Tensor, Y: torch.Tensor, data_range=255., size_averaged: bool = True):
    """SSIM评价标准
    计算预测张量序列和标签张量序列中，每组张量的SSIM
    SSIM值通过torchmetrics包计算
    可通过设置size_averaged来进行批量平均

    :param Y_HAT: 预测张量序列
    :param Y: 标签张量序列
    :param data_range: 张量的取值范围。
        若是灰度图或RGB图，则取值为255；
        若是二值图，则取值为1
    :param size_averaged: 是否进行批量平均
    :return: 计算结果张量
    """
    if size_averaged:
        kwargs = {'data_range': data_range, 'reduction': 'sum'}
    else:
        kwargs = {'data_range': data_range, 'reduction': 'none'}
    computer = StructuralSimilarityIndexMeasure(**kwargs).to(Y_HAT.device)
    return computer(Y_HAT, Y)


def PSNR(Y_HAT: torch.Tensor, Y: torch.Tensor, data_range=255., size_averaged: bool = True):
    """PSNR评价标准
    计算预测张量序列和标签张量序列中，每组张量的PSNR
    PSNR值通过torchmetrics包计算
    可通过设置size_averaged来进行批量平均

    :param Y_HAT: 预测张量序列
    :param Y: 标签张量序列
    :param data_range: 张量的取值范围。
        若是灰度图或RGB图，则取值为255；
        若是二值图，则取值为1
    :param size_averaged: 是否进行批量平均
    :return: 计算结果张量
    """
    if size_averaged:
        kwargs = {'data_range': data_range, 'reduction': 'sum', 'dim': list(range(1, len(Y_HAT.shape)))}
    else:
        kwargs = {'data_range': data_range, 'reduction': 'none', 'dim': list(range(1, len(Y_HAT.shape)))}
    computer = PeakSignalNoiseRatio(**kwargs).to(Y_HAT.device)
    return computer(Y_HAT, Y)


def PCC(Y_HAT: torch.Tensor, Y: torch.Tensor, size_averaged: bool = True):
    """PCC评价标准
    计算预测张量序列和标签张量序列中，每组张量的PCC
    PCC值通过torch.corrcoef()计算
    可通过设置size_averaged来进行批量平均

    :param Y_HAT: 预测张量序列
    :param Y: 标签张量序列
    :param size_averaged: 是否进行批量平均
    :return: 计算结果张量
    """
    from layers import PCC as PearsonCorrCoef

    if size_averaged:
        kwargs = {'reduction': 'sum'}
    else:
        kwargs = {'reduction': 'none'}
    computer = PearsonCorrCoef(**kwargs)
    return computer(Y_HAT, Y)
