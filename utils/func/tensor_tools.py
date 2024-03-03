from typing import List, Iterable

import torch
import torchvision
from PIL import Image as IMAGE
from PIL.Image import Image

import numpy as np


img_modes = ['L', 'RGB', '1']


def __ts_to_BiLevelImg(ts: torch.Tensor) -> Image:
    assert len(ts.shape) == 3, f'本方法只接受三维输入，输入的张量形状为{ts.shape}'
    # 不转换为灰度值则图片转换后变为全黑
    ts = ts.reshape(ts.shape[1:]) * 255
    ts = ts.cpu().numpy()
    return IMAGE.fromarray(ts).convert('1')


def tensor_to_img(ts: torch.Tensor, mode: str = 'RGB') -> List[Image]:
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    assert len(ts.shape) == 4, f'本方法只接受四维输入（批量大小，通道数，长，宽），输入的张量形状为{ts.shape}'
    ts = ts.cpu()
    ret = []
    if mode == '1':
        for t in ts:
            ret.append(__ts_to_BiLevelImg(t))
    else:
        for t in ts:
            ret.append(torchvision.transforms.ToPILImage(mode)(t))
    return ret


def img_to_tensor(imgs: List[Image],
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device('cpu')) -> torch.Tensor:
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
