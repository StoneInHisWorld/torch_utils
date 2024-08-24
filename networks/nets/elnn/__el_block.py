from networks import BasicNN
from torch.nn import Conv2d, ConvTranspose2d


class ELBlockGenerator:

    def __init__(self, version, norm_layer, activation):
        self.version = version
        self.norm_layer = norm_layer
        self.activation = activation

        if version == 'v1' or version == 'v2':
            def get_blocks(i, o, cur_shape, required_shape):
                ct_s, ct_k, ct_p = 2, 3, 1
                # oshape_h = (cur_shape[0] - 1) * ct_s + ct_k - 2 * ct_p
                # oshape_w = (cur_shape[0] - 1) * ct_s + ct_k - 2 * ct_p
                if self.whether_enlarge(ct_s, ct_k, ct_p, cur_shape, required_shape):
                    return (cur_shape[0] * 2 - 1, cur_shape[1] * 2 - 1), [
                        # 提取特征
                        Conv2d(i, o, kernel_size=3, stride=1, padding=1),
                        self.norm_layer(o),
                        self.activation(),
                        # 上采样2倍
                        ConvTranspose2d(o, o, kernel_size=ct_k, stride=ct_s, padding=ct_p),
                        self.norm_layer(o),
                        self.activation()
                    ]
                else:
                    return cur_shape, [
                        # 提取特征
                        Conv2d(i, o, kernel_size=3, stride=1, padding=1),
                        self.norm_layer(o),
                        self.activation(),
                    ]
        else:
            raise NotImplementedError(f'无法识别的版本号: {version}！')

        self.get_blocks = get_blocks

    def whether_enlarge(self, ct_s, ct_k, ct_p, cur_shape, required_shape):
        oshape_h = (cur_shape[0] - 1) * ct_s + ct_k - 2 * ct_p
        oshape_w = (cur_shape[0] - 1) * ct_s + ct_k - 2 * ct_p
        return oshape_h < required_shape[0] and oshape_w < required_shape[1]
