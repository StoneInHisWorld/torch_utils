import torch
from torch import nn
from torchvision.transforms import Resize


class Reshape(nn.Module):

    def __init__(self, required_shape):
        """
        重塑层，可以将输入的张量进行重塑为所需形状。
        :param required_shape: 重塑目标形状
        """
        super().__init__()
        self.required_shape = required_shape

    def forward(self, data: torch.Tensor):
        # 实现将图片进行随机裁剪以达到目标shape的功能
        dl = data.shape[0]
        ih, iw = data.shape[-2:]
        h, w = self.required_shape

        # 长边放缩比例
        scale = max(w / iw, h / ih)
        # 计算新图片shape
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        # 计算图片缺失shape
        dw = w - new_w
        dh = h - new_h
        # 等比例缩放数据
        resizer = Resize((new_h, new_w), antialias=True)
        data = resizer(data)
        # 若需求图片大小较大，则进行填充
        if dw > 0:
            data = nn.ReflectionPad2d((0, dw, 0, 0))(data)
        if dh > 0:
            data = nn.ReflectionPad2d((0, 0, 0, dh))(data)
        # 若需求图片大小较小，则随机取部分
        if dw < 0 or dh < 0:
            new_data = []
            rand_w = torch.randint(0, abs(dw), (dl,)) if dw < 0 else 0
            rand_h = torch.randint(0, abs(dh), (dl,)) if dh < 0 else 0
            for i, _ in enumerate(data):
                i_h = rand_h[i] if dh < 0 else 0
                i_w = rand_w[i] if dw < 0 else 0
                new_data.append(
                    data[i, :, i_h: i_h + h, i_w: i_w + w].reshape((1, -1, h, w))
                )
            data = torch.vstack(new_data)
        return data
