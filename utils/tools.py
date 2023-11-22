import os.path
import random
from typing import Tuple

import pandas as pd
import torch
from PIL import Image as IMAGE
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import cuda, nn as nn
from torch.nn import init as init

from networks.layers import common_layers

optimizers = ['sgd', 'adam']
loss_es = ['l1', 'entro', 'mse', 'huber', 'ssim']
init_funcs = ['normal', 'xavier', 'zero']


def write_log(path: str, **kwargs):
    assert path.endswith('.csv'), f'日志文件格式为.csv，将要写入的文件名为{path}'
    try:
        file_data = pd.read_csv(path)
    except Exception as _:
        file_data = pd.DataFrame([])
    item_data = pd.DataFrame([kwargs])
    if len(file_data) == 0:
        file_data = pd.DataFrame(item_data)
    else:
        file_data = pd.concat([file_data, item_data], axis=0, sort=True)
    file_data.to_csv(path, index=False)


def plot_history(history, mute=False, title=None, xlabel=None,
                 ylabel=None, savefig_as=None, accumulative=False):
    # check_path(savefig_as)
    for label, log in history:
        plt.plot(range(len(log)), log, label=label)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if savefig_as:
        if not os.path.exists(os.path.split(savefig_as)[0]):
            os.makedirs(os.path.split(savefig_as)[0])
        plt.savefig(savefig_as)
    plt.legend()
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
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            momentum=momentum
        )
    elif optim_str == 'adam':
        return torch.optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
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
        return common_layers.SSIMLoss()
    

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
    else:
        w_init, b_init = init.zeros_, init.zeros_

    def _init(m: nn.Module) -> None:
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            w_init(m.weight)
            b_init(m.bias)

    return _init

# def resize_img(image: Image, required_shape: Tuple[int, int], img_mode='L') -> Image:
#     # ------------------------------#
#     #   获得图像的高宽与目标高宽
#     #   code from: Bubbliiiing
#     # ------------------------------#
#     iw, ih = image.size
#     h, w = required_shape
#
#     # 长边放缩比例
#     scale = min(w / iw, h / ih)
#     # 计算新图片shape
#     new_w = int(iw * scale)
#     new_h = int(ih * scale)
#     # 计算图片缺失shape
#     dx = (w - new_w) // 2
#     dy = (h - new_h) // 2
#     # ---------------------------------#
#     #   将图像多余的部分加上黑条
#     # ---------------------------------#
#     image = image.resize((new_w, new_h), IMAGE.BICUBIC)
#     new_image = IMAGE.new(img_mode, (w, h), 0)
#     new_image.paste(image, (dx, dy))
#
#     return new_image


def resize_img(image: Image, required_shape: Tuple[int, int], img_mode='L') -> Image:
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
        back_ground = IMAGE.new(img_mode, (w, h), 0)
        back_ground.paste(image)
    # 若需求图片大小较小，则随机取部分
    if dw < 0 or dh < 0:
        i_h = random.randint(0, -dh) if dh < 0 else 0
        i_w = random.randint(0, -dw) if dw < 0 else 0
        image.crop((i_w, i_w, i_w + new_w, i_h + new_h))
    return image


def check_path(path: str, way_to_mkfile=None):
    """
    检查指定路径。如果目录不存在，则会创建目录；如果文件不存在，则指定文件初始化方式后才会自动初始化文件
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
