import torch
import torchvision
from PIL import Image as IMAGE
from PIL.Image import Image

import numpy as np


img_modes = ['L', 'RGB', '1']


def tensor_to_img(ts: torch.Tensor, mode: str = 'RGB') -> Image:
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    if mode == '1':
        ts = ts.cpu().numpy()
        ts = ts.reshape(ts.shape[1:]) * 255
        ret = IMAGE.fromarray(ts).convert(mode)
    else:
        ret = torchvision.transforms.ToPILImage(mode)(ts)
    return ret.convert(mode)


def img_to_tensor(img: Image,
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device('cpu')) -> torch.Tensor:
    mode = img.mode
    img = np.array(img)
    if mode == '1' or mode == 'L':
        img = img.reshape((1, *img.shape[:2]))
    else:
        img = img.reshape((3, *img.shape[:2]))
    img = torch.from_numpy(img).to(dtype)
    return img.to(device)
