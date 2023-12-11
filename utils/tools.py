import os.path
import random
import warnings
from typing import Tuple

import pandas as pd
import torch
from PIL import Image as IMAGE
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import cuda, nn as nn
from torch.nn import init as init

from networks.layers import ssim as cl

optimizers = ['sgd', 'asgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax']
loss_es = ['l1', 'entro', 'mse', 'huber', 'ssim']
# init_funcs = ['normal', 'xavier', 'zero', 'entire_trained', 'state_trained']
init_funcs = ['normal', 'xavier', 'zero']


def write_log(path: str, **kwargs):
    """
    编写运行日志。
    :param path: 日志保存路径
    :param kwargs: 日志保存信息，类型为词典，key为列名，value为单元格内容
    :return: None
    """
    assert path.endswith('.csv'), f'日志文件格式为.csv，但指定的文件名为{path}'
    try:
        file_data = pd.read_csv(path, encoding='utf-8')
    except Exception as _:
        file_data = pd.DataFrame([])
    item_data = pd.DataFrame([kwargs])
    if len(file_data) == 0:
        file_data = pd.DataFrame(item_data)
    else:
        file_data = pd.concat([file_data, item_data], axis=0, sort=True)
    file_data.to_csv(path, index=False, encoding='utf-8-sig')


# def plot_history(history, mute=False, title=None, xlabel=None,
#                  ylabel=None, savefig_as=None, accumulative=False):
#     """
#     绘制训练历史变化趋势图
#     :param history: 训练历史数据
#     :param mute: 绘制完毕后是否立即展示成果图
#     :param title: 绘制图标题
#     :param xlabel: 自变量名称
#     :param ylabel: 因变量名称
#     :param savefig_as: 保存图片路径
#     :param accumulative: 是否将所有趋势图叠加在一起
#     :return: None
#     """
#     warnings.warn('将在未来版本中删除！', DeprecationWarning)
#     for label, log in history:
#         plt.plot(range(len(log)), log, label=label)
#     if xlabel:
#         plt.xlabel(xlabel)
#     if ylabel:
#         plt.ylabel(ylabel)
#     if title:
#         plt.title(title)
#     plt.legend()
#     if savefig_as:
#         if not os.path.exists(os.path.split(savefig_as)[0]):
#             os.makedirs(os.path.split(savefig_as)[0])
#         plt.savefig(savefig_as)
#         print('已保存历史趋势图')
#     if not mute:
#         plt.show()
#     if not accumulative:
#         plt.clf()

def plot_history(history, mute=False, title=None, ls_ylabel=None,
                 acc_ylabel=None, savefig_as=None, accumulative=False):
    """
    绘制训练历史变化趋势图
    :param history: 训练历史数据
    :param mute: 绘制完毕后是否立即展示成果图
    :param title: 绘制图标题
    :param xlabel: 自变量名称
    :param ylabel: 因变量名称
    :param savefig_as: 保存图片路径
    :param accumulative: 是否将所有趋势图叠加在一起
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(7, 6))
    ax1.set_title('LOSS')
    ax2.set_xlabel('epochs')
    ax2.set_title('ACCURACY')
    for label, log in history:
        if label.find('_l') != -1:
            # 绘制损失值history
            ax1.plot(range(1, len(log) + 1), log, label=label)
        elif label.find('_acc') != -1:
            # 绘制准确率history
            ax2.plot(range(1, len(log) + 1), log, label=label)
    if ls_ylabel:
        ax1.set_ylabel(ls_ylabel)
    if acc_ylabel:
        ax2.set_ylabel(acc_ylabel)
    if title:
        fig.suptitle(title)
    ax1.legend()
    ax2.legend()
    if savefig_as:
        if not os.path.exists(os.path.split(savefig_as)[0]):
            os.makedirs(os.path.split(savefig_as)[0])
        plt.savefig(savefig_as)
        print('已保存历史趋势图')
    if not mute:
        plt.show()
    if not accumulative:
        plt.clf()


def try_gpu(i=0):
    """
    获取一个GPU
    :param i: GPU编号
    :return: 第i号GPU。若GPU不可用，则返回CPU
    """
    if cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def permutation(res: list, *args):
    """
    生成超参数列表。
    :param res: 结果列表。每个输出的列表都以`res`为前缀。
    :param args: 超参数列表。每个超参数输入均为列表，列表中的值为该超参数可能取值
    :return: 超参数取值列表生成器。
    """
    if len(args) == 0:
        yield res
    elif len(args) == 1:
        for arg in args[0]:
            yield res + [arg]
    else:
        for arg in args[0]:
            for p in permutation(res + [arg], *args[1:]):  # 不能直接yield本函数，否则不会执行
                yield p


def get_optimizer(net: torch.nn.Module, optim_str, lr=0.1, w_decay=0., momentum=0.):
    assert optim_str in optimizers, f'不支持优化器{optim_str}, 支持的优化器包括{optimizers}'
    if optim_str == 'sgd':
        # 使用随机梯度下降优化器
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            momentum=momentum
        )
    elif optim_str == 'asgd':
        # 使用随机平均梯度下降优化器
        return torch.optim.ASGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'adagrad':
        # 使用自适应梯度优化器
        return torch.optim.Adagrad(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'adadelta':
        # 使用Adadelta优化器，Adadelta是Adagrad的改进
        return torch.optim.Adadelta(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'rmsprop':
        # 使用RMSprop优化器，RMSprop是Adagrad的改进
        return torch.optim.RMSprop(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            momentum=momentum
        )
    elif optim_str == 'adam':
        # 使用Adaptive Moment Estimation优化器。Adam是RMSprop的改进。
        return torch.optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'adamax':
        # 使用Adamax优化器，Adamax是Adam的改进
        return torch.optim.Adamax(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
        )


def get_loss(loss_str: str = 'mse'):
    """
    获取损失函数。
    :param loss_str: 损失函数对应字符串
    :return: 损失函数模块
    """
    assert loss_str in loss_es, \
        f'不支持损失函数{loss_str}, 支持的损失函数包括{loss_es}'
    if loss_str == 'l1':
        return nn.L1Loss()
    elif loss_str == 'entro':
        return nn.CrossEntropyLoss()
    elif loss_str == 'mse':
        return nn.MSELoss()
    elif loss_str == 'huber':
        return nn.HuberLoss()
    elif loss_str == 'ssim':
        return cl.SSIMLoss()


def init_wb(func_str: str = 'xavier'):
    """
    返回初始化权重、偏移参数的函数。
    :param func_str: 指定初始化方法的字符串
    :return: 包装好可直接调用的初始化函数
    """
    assert func_str in init_funcs, f'不支持的初始化方式{func_str}, 当前支持的初始化方式包括{init_funcs}'
    if func_str == 'normal':
        w_init = lambda m: init.normal_(m, 0, 1)
        b_init = lambda m: init.normal_(m, 0, 1)
    elif func_str == 'xavier':
        w_init, b_init = init.xavier_uniform_, init.zeros_
    # elif func_str == 'entire_trained':
    #     pass
    # elif func_str == 'state_trained':
    #     def _init(m: nn.Module) -> None:
    #         sd = torch.load(where)
    else:
        w_init, b_init = init.zeros_, init.zeros_

    def _init(m: nn.Module) -> None:
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            w_init(m.weight)
            b_init(m.bias)

    return _init


def resize_img(image: Image, required_shape: Tuple[int, int]) -> Image:
    """
    重塑图片。
    先将图片等比例放大到最大（放缩到最小）满足required_shape的尺寸，再对图片随机取部分或填充黑边以适配所需形状
    :param image: 待编辑图片
    :param required_shape: 所需形状
    :return: 重塑完成的图片。
    """
    # 实现将图片进行随机裁剪以达到目标shape的功能
    ih, iw = image.size
    h, w = required_shape

    # 长边放缩比例
    scale = max(w / iw, h / ih)
    # 计算新图片shape
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    # 计算图片缺失shape
    dw = w - new_w
    dh = h - new_h
    # 等比例缩放数据
    image = image.resize((new_h, new_w), IMAGE.BICUBIC)
    # 若需求图片大小较大，则进行填充
    if dw > 0 or dh > 0:
        back_ground = IMAGE.new(image.mode, (w, h), 0)
        back_ground.paste(image)
    # 若需求图片大小较小，则随机取部分
    if dw < 0 or dh < 0:
        i_h = random.randint(0, -dh) if dh < 0 else 0
        i_w = random.randint(0, -dw) if dw < 0 else 0
        image.crop((i_w, i_w, i_w + new_w, i_h + new_h))
    return image


def crop_img(img: Image, required_shape, loc: str or Tuple[int, int]) -> Image:
    """
    按照指定位置裁剪图片
    :param img: 即将进行裁剪的图片
    :param required_shape: 需要保留的尺寸
    :param loc: 裁剪的位置。可以指定为“lt, lb, rt, rb, c”的其中一种，或者指定为二元组指示裁剪区域的左上角坐标
    :return: 裁剪完成的图片
    """
    img_size = img.size
    ih, iw = img_size
    rh, rw = required_shape
    assert rh <= ih and rw <= iw, (
        f'裁剪尺寸{required_shape}需要小于图片尺寸{img_size}！'
    )
    if type(loc) == str:
        if loc == 'lt':
            loc = (0, 0, 0 + rw, 0 + rh)
        elif loc == 'lb':
            loc = (0, ih - rh, 0 + rw, iw)
        elif loc == 'rt':
            loc = (iw - rw, 0, iw, rh)
        elif loc == 'rb':
            loc = (iw - rw, ih - rh, ih, iw)
        elif loc == 'c':
            loc = (iw // 2 - rw // 2, ih // 2 - rh // 2, iw // 2 + rw // 2, ih // 2 + rh // 2)
        else:
            raise Exception(f'不支持的裁剪位置{loc}！')
    elif type(loc) == Tuple and len(loc) == 2:
        loc = (loc[0], loc[1], loc[0] + rw, loc[1] + rh)
    else:
        raise Exception(f'无法识别的裁剪位置参数{loc}')
    return img.crop(loc)


def check_path(path: str, way_to_mkfile=None):
    """
    检查指定路径。如果目录不存在，则会创建目录；如果文件不存在，则指定文件初始化方式后才会自动初始化文件
    :param path: 需要检查的目录
    :param way_to_mkfile: 初始化文件的方法
    :return: None
    """
    if not os.path.exists(path):
        path, file = os.path.split(path)
        if file != "":
            # 如果是文件
            if way_to_mkfile is not None:
                # 如果指定了文件初始化方式，则自动初始化文件
                if path == "" or os.path.exists(path):
                    way_to_mkfile(file)
                else:
                    os.makedirs(path)
                    way_to_mkfile(os.path.join(path, file))
            else:
                raise FileNotFoundError(f'没有在{path}下找到{file}文件！')
        else:
            # 如果目录不存在，则新建目录
            os.makedirs(path)


def check_para(name, value, val_range) -> bool:
    if value in val_range:
        return True
    else:
        warnings.warn(f'参数{name}需要取值限于{val_range}！')
        return False


def get_logData(log_path, exp_no) -> dict:
    log = pd.read_csv(log_path)
    log = log.set_index('exp_no').to_dict('index')
    return log[exp_no]
