import os
import random
from typing import Tuple, Iterable, Callable

import numpy as np
from PIL import Image as IMAGE, ImageDraw
from PIL.Image import Image


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


def read_img(path: str, mode: str = 'L', requires_id: bool = False,
             *preprocesses: Iterable[Tuple[Callable, dict]]) -> np.ndarray:
    """
    读取图片
    :param path: 图片所在路径
    :param mode: 图片读取模式
    :param requires_id: 是否需要给图片打上ID
    :return: 图片对应numpy数组，形状为（通道，图片高，图片宽，……）
    """
    img_modes = ['L', 'RGB', '1']
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    # img = Image.open(path).convert(mode)
    img = IMAGE.open(path)
    preprocesses = (*preprocesses, (Image.convert, (mode, ), {}))
    for preprocess in preprocesses:
        func, args, kwargs = preprocess
        img = func(img, *args, **kwargs)
    img = np.array(img)
    # 复原出通道。1表示样本数量维
    if mode == 'L' or mode == '1':
        img_channels = 1
    elif mode == 'RGB':
        img_channels = 3
    else:
        img_channels = -1
    img = img.reshape((img_channels, *img.shape[:2]))
    if requires_id:
        # 添加上读取文件名
        file_name = os.path.split(path)[-1]
        img = np.hstack((file_name, img))
    return img


def binarize_img(img: Image, threshold: int = 127) -> Image:
    """
    将图片根据阈值进行二值化
    参考自：https://www.jianshu.com/p/f6d40a73310f
    :param img: 待转换图片
    :param threshold: 二值图阈值
    :return: 转换好的图片
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    # 图片二值化
    return img.point(table, '1')


def concat_imgs(comment="", *imgs_and_labels: Tuple[Image, str]) -> Image:
    text_size = 15
    border = 5
    # # TODO：如何判断白板所用的图片模式？
    img_mode = imgs_and_labels[0][0].mode
    wb_width = (len(imgs_and_labels) + 1) * border + sum(
        [img.width for img, _ in imgs_and_labels]
    )
    wb_height = 4 * border + 2 * text_size + max(
        [img.height for img, _ in imgs_and_labels]
    )  # 留出一栏填充comment
    # 制作输入、输出、标签对照图
    whiteboard = IMAGE.new(
        img_mode, (wb_width, wb_height), color=255
    )
    draw = ImageDraw.Draw(whiteboard)
    # 绘制标签
    for i in range(len(imgs_and_labels)):
        draw.text(
            (
                (i + 1) * border + sum([img.width for img, _ in imgs_and_labels[: i]]),
                border
            ),
            imgs_and_labels[i][1]
        )
    # 粘贴图片
    for i, (img, label) in enumerate(imgs_and_labels):
        whiteboard.paste(img,
                         ((i + 1) * border + sum([img.width for img, _ in imgs_and_labels[: i]]),
                          border + text_size))
    # 绘制脚注
    draw.text(
        (border, wb_height - text_size - border), 'COMMENT: ' + comment
    )
    return whiteboard
