from torch.nn import Conv2d


class OutputBlockGenerator:

    def __init__(self, version, norm_layer, activation):
        self.version = version
        self.norm_layer = norm_layer
        self.activation = activation

    def get_blocks(self, i, o):
        if self.version == 'v1':
            return [
                # 压缩特征
                Conv2d(i, o, kernel_size=3, stride=1, padding=1),
                self.norm_layer(o),
                self.activation(),
            ]
        elif self.version == 'v2':
            return [
                # 压缩特征
                Conv2d(i, o, kernel_size=1),
                self.norm_layer(o),
                self.activation(),
            ]
        else:
            raise ValueError(f"不支持的版本{self.version}")
