import torch
from torch import nn
from typing import List

import utils.tools as tools
from torchvision.transforms import Resize


class Reshape(nn.Module):

    def __init__(self, required_shape):
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


class DualOutputLayer(nn.Module):

    def __init__(self, in_features, fir_out, sec_out, dropout_rate=0.,
                 momentum=0.) -> None:
        super().__init__()
        fir = nn.Sequential(
            *self.__get_layers__(in_features, fir_out, dropout=dropout_rate,
                                 momentum=momentum)
        )
        sec = nn.Sequential(
            *self.__get_layers__(in_features, sec_out, dropout=dropout_rate,
                                 momentum=momentum)
        )
        fir.apply(tools.init_wb)
        sec.apply(tools.init_wb)
        self.add_module('fir', fir)
        self.add_module('sec', sec)

    def forward(self, features):
        in_features_es = [
            child[1].in_features for _, child in self
        ]
        batch_size = len(features)
        feature_batch_es = [
            features.reshape((batch_size, in_fea))
            for in_fea in in_features_es
        ]
        fir_out = self[0](feature_batch_es[0])
        sec_out = self[1](feature_batch_es[1])
        return torch.hstack((fir_out, sec_out))

    def __get_layers__(self, in_features: int, out_features: int, dropout=0.,
                       momentum=0.) -> List[nn.Module]:
        assert in_features > 0 and out_features > 0, '输入维度与输出维度均需大于0'
        # layers = [nn.BatchNorm1d(in_features)]
        # # 构造一个三层感知机
        # trans_layer_sizes = [in_features, (in_features + out_features) // 2, out_features]
        # # 对于每个layer_size加入全连接层、BN层以及Dropout层
        # for i in range(len(trans_layer_sizes) - 1):
        #     in_size, out_size = trans_layer_sizes[i], trans_layer_sizes[i + 1]
        #     layers += [
        #         nn.Linear(in_size, out_size),
        #         # nn.BatchNorm1d(out_size, momentum=momentum),
        #         nn.LeakyReLU()
        #     ]
        #     if dropout > 0.:
        #         layers.append(nn.Dropout())
        # # 去掉最后的Dropout层
        # if type(layers[-1]) == nn.Dropout:
        #     layers.pop(-1)
        # # 将最后的激活函数换成Softmax
        # layers.pop(-1)
        # 加入Softmax层
        layers = [
            nn.Flatten(),
            nn.Linear(in_features, out_features)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Softmax(dim=1))
        return layers

    def __iter__(self):
        return self.named_children()

    def __getitem__(self, item: int):
        children = self.named_children()
        for _ in range(item):
            next(children)
        return next(children)[1]


