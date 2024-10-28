import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from networks.layers import PCC as PearsonCorrCoef


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
    #
    # with warnings.catch_warnings(record=True) as warning_filter:
    # warnings.simplefilter("ignore", UserWarning)
    computer = StructuralSimilarityIndexMeasure(**kwargs).to(Y_HAT.device)
    return computer(Y_HAT, Y)
        # def msg_printer(*input_args):
        #     constant_msg = ''
        #     y, y_hat = input_args
        #     constant_msg += f'Y:{y[0]}' if np.all(y == y[0]) else ''
        #     constant_msg += f'Y_HAT:{y_hat[0]}' if np.all(y_hat == y_hat[0]) else ''
        #     print(f'\n计算皮尔逊相关系数时遇到常数输入，此时相关系数无定义！常数输入如下：\n{constant_msg}')
        #
        # pearsonr_impl = functools.partial(
        #     warning_handler, func=lambda y, y_hat: PearsonCorrCoef(y, y_hat)[0],
        #     category=ConstantInputWarning, warning_filter=warning_filter, warning_msg_printer=msg_printer
        # )

    # y = Y.cpu().numpy().squeeze()
    # y_hat = Y_HAT.cpu().numpy().squeeze()
    # if len(y.shape) == 2 and len(y_hat.shape) == 2:
    #     # 如果只有一张图片
    #     result = torch.tensor(ssim(y, y_hat, channel_axis=0, data_range=1))
    # else:
    #     result = torch.tensor([ssim(i, j, channel_axis=0, data_range=1) for i, j in zip(y, y_hat)])
    # if size_averaged:
    #     return torch.sum(result)
    # else:
    #     return result


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
    # with warnings.catch_warnings(record=True) as warning_filter:
    #     warnings.simplefilter("ignore", UserWarning)
    computer = PeakSignalNoiseRatio(**kwargs).to(Y_HAT.device)
    return computer(Y_HAT, Y)

    # y = normalize(Y).cpu().numpy()
    # y_hat = normalize(Y_HAT).cpu().numpy()
    # if len(y.shape) == 2 and len(y_hat.shape) == 2:
    #     # 如果只有一张图片
    #     result = torch.tensor(psnr(y, y_hat, data_range=1))
    # else:
    #     result = torch.tensor([psnr(i, j, data_range=1) for i, j in zip(y, y_hat)])
    # if size_averaged:
    #     return torch.sum(result)
    # else:
    #     return result


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
    if size_averaged:
        kwargs = {'reduction': 'sum'}
    else:
        kwargs = {'reduction': 'none'}
    computer = PearsonCorrCoef(**kwargs)
    return computer(Y_HAT, Y)
#     with warnings.catch_warnings(record=True) as warning_filter:
#         warnings.simplefilter("default", ConstantInputWarning)
#
#         def msg_printer(*input_args):
#             constant_msg = ''
#             y, y_hat = input_args
#             constant_msg += f'Y:{y[0]}' if np.all(y == y[0]) else ''
#             constant_msg += f'Y_HAT:{y_hat[0]}' if np.all(y_hat == y_hat[0]) else ''
#             print(f'\n计算皮尔逊相关系数时遇到常数输入，此时相关系数无定义！常数输入如下：\n{constant_msg}')
#
#         pearsonr_impl = functools.partial(
#             warning_handler, func=lambda y, y_hat: PearsonCorrCoef(y, y_hat)[0],
#             category=ConstantInputWarning, warning_filter=warning_filter, warning_msg_printer=msg_printer
#         )
#         if len(y.shape) == 2 and len(y_hat.shape) == 2:
#             # 如果只有一张图片
#             result = torch.tensor(pearsonr_impl(y.flatten(), y_hat.flatten()))
#         else:
#             result = torch.tensor([pearsonr_impl(i.flatten(), j.flatten()) for i, j in zip(y, y_hat)])
#     if size_averaged:
#         return torch.sum(result)
#     else:
#         return result
#     computer = PearsonCorrCoef(**kwargs)
#     return computer(Y_HAT, Y)

    # y = normalize(Y).cpu().numpy()
    # y_hat = normalize(Y_HAT).cpu().numpy()
    # with warnings.catch_warnings(record=True) as warning_filter:
    #     warnings.simplefilter("default", ConstantInputWarning)
    #
    #     def msg_printer(*input_args):
    #         constant_msg = ''
    #         y, y_hat = input_args
    #         constant_msg += f'Y:{y[0]}' if np.all(y == y[0]) else ''
    #         constant_msg += f'Y_HAT:{y_hat[0]}' if np.all(y_hat == y_hat[0]) else ''
    #         print(f'\n计算皮尔逊相关系数时遇到常数输入，此时相关系数无定义！常数输入如下：\n{constant_msg}')
    #
    #     pearsonr_impl = functools.partial(
    #         warning_handler, func=lambda y, y_hat: pearsonr(y, y_hat)[0],
    #         category=ConstantInputWarning, warning_filter=warning_filter, warning_msg_printer=msg_printer
    #     )
    #     if len(y.shape) == 2 and len(y_hat.shape) == 2:
    #         # 如果只有一张图片
    #         result = torch.tensor(pearsonr_impl(y.flatten(), y_hat.flatten()))
    #     else:
    #         result = torch.tensor([pearsonr_impl(i.flatten(), j.flatten()) for i, j in zip(y, y_hat)])
    # if size_averaged:
    #     return torch.sum(result)
    # else:
    #     return result
