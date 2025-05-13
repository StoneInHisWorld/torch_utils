import torch
import torch.nn as nn


class Patching2D(nn.Module):

    def __init__(self, patch_size, flatten=True):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, "Patching2D只接受二维图片输入"
        self.patch_size = patch_size
        self.flatten = flatten
        super().__init__()

    def forward(self, inputs):
        assert len(inputs.shape) == 4, \
            f"Patching2D需要输入的图片维度是（批量大小, 通道, 高度, 宽度），但接收到的输入是{inputs.shape}"
        # 展开空间维度 (B, C, H, W) → (B, C, num_h, pH, num_w, pW)
        patches = inputs.unfold(2, *self.patch_size).unfold(3, *self.patch_size)
        # 调整维度顺序 → (B, num_h, num_w, C, pH, pW)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # 合并块维度 → (B, num_h*num_w, C, pH, pW)
        if self.flatten:
            # 二维图片块
            patches = patches.reshape([*patches.shape[:3], -1])
            return patches.reshape([patches.shape[0], -1, patches.shape[-1]])
        else:
            return patches