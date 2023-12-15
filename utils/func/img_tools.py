import random
from typing import Tuple

from PIL import Image as IMAGE
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
