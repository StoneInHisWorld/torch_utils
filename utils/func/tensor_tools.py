from typing import List, Iterable

import numpy as np
import torch
import torchvision
from PIL import Image as IMAGE
from PIL.Image import Image

img_modes = ['L', 'RGB', '1']


def __ts_to_BiLevelImg(ts: torch.Tensor) -> Image:
    """将单张三维张量转换为二值（1-bit）PIL 图像。

    :param ts: 单张图片张量，形状应为 (C, H, W)。
    :return: 转换后的二值 PIL 图片对象。
    """
    assert len(ts.shape) == 3, f'本方法只接受三维输入，输入的张量形状为{ts.shape}'
    # 不转换为灰度值则图片转换后变为全黑
    ts = ts.reshape(ts.shape[1:]) * 255
    ts = ts.cpu().numpy()
    return IMAGE.fromarray(ts).convert('1')


def tensor_to_img(ts: torch.Tensor, mode: str = 'RGB') -> List[Image]:
    """将四维批量张量转换为指定模式的 PIL 图像列表。

    :param ts: 批量图片张量，形状应为 (N, C, H, W)。
    :param mode: 输出图像模式，支持 'L'、'RGB'、'1'。
    :return: 转换后的 PIL 图片列表。
    """
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    assert len(ts.shape) == 4, f'本方法只接受四维输入（批量大小，通道数，长，宽），输入的张量形状为{ts.shape}'
    ts = ts.cpu()
    ret = []
    if mode == '1':
        for t in ts:
            ret.append(__ts_to_BiLevelImg(t))
    else:
        for t in ts:
            # PIL图片要求数据格式为uint8，否则其他格式的张量会出现偏色的问题
            t = t.type(torch.uint8)
            ret.append(torchvision.transforms.ToPILImage()(t))
    return ret


def img_to_tensor(imgs: List[Image],
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """将 PIL 图像序列转换为批量张量并放置到指定设备。

    :param imgs: PIL 图片序列，序列内图片模式需一致。
    :param dtype: 输出张量的数据类型。
    :param device: 输出张量所在设备。
    :return: 形状为 (N, C, H, W) 的批量张量。
    """
    assert isinstance(imgs, Iterable), f'本方法只接受图片序列输入！'
    mode = imgs[0].mode
    ts = []
    for img in imgs:
        img = np.array(img)
        if mode == '1' or mode == 'L':
            img = img.reshape((1, *img.shape[:2]))
        else:
            img = img.reshape((3, *img.shape[:2]))
        ts.append(img)
    return torch.tensor(np.array(ts), dtype=dtype, device=device)
