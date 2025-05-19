import torch
from torch import nn

_supported_position_Emb = ['bert', 'original']


class AddPositionEmbs(nn.Module):
    """为输入张量添加位置编码
    该层权重不支持自定义初始化

    Attributes:
        which: 使用哪种位置编码
        inputs_size: 输入张量的形状
        dropout: original模式下，Dropout概率
        device: 权重参数所处设备
        dtype: 权重参数数据类型
    """

    def __init__(self, which, inputs_size, dropout=0.1,
                 device=torch.device('cpu'), dtype=torch.float32):
        """为输入张量添加位置编码

        Args:
            which: 使用哪种位置编码
            inputs_size: 输入张量的形状
            dropout: original模式下，Dropout概率
            device: 权重参数所处设备
            dtype: 权重参数数据类型
        """
        super(AddPositionEmbs, self).__init__()
        assert len(inputs_size) == 2, "只接受三维向量输入"
        self.input_size = inputs_size
        assert which in _supported_position_Emb, \
            f"不识别的位置编码模式{which}！支持的模式包括：{_supported_position_Emb}"
        if which == 'bert':
            self.weights = self.bert_init(inputs_size, device, dtype)
            self._forward_impl = self.bert_forward
        elif which == "original":
            self.dropout = nn.Dropout(dropout)
            self.weights = self.original_init(inputs_size, device, dtype)
            self._forward_impl = self.original_forward

    def forward(self, inputs):
        return self._forward_impl(inputs)

    def to(self, device):
        """将权重参数转移到指定设备上

        Args:
            device: 权重参数所处设备
        """
        super(AddPositionEmbs, self).to(device)
        self.weights = self.weights.to(device)
        return self

    def bert_forward(self, inputs):
        """BERT模式下的添加位置编码

        Args:
            inputs: 前向传播的输入

        Returns:
            输出为（批量大小, 时序步数, 输入维度）
        """
        assert inputs.shape[1:] == self.input_size, \
            (f"期待的输入形状为{("batch_size", *self.input_size)}，"
             f"而实际上得到的输入形状为{inputs.shape}")
        return inputs + self.weights

    def original_forward(self, inputs):
        """原始Transformer模式下的添加位置编码

        Args:
            inputs: 前向传播的输入

        Returns:
            输出为（批量大小, 时序步数, 输入维度）
        """
        assert inputs.shape[1:] == self.input_size, \
            (f"期待的输入形状为{("batch_size", *self.input_size)}，"
             f"而实际上得到的输入形状为{inputs.shape}")
        inputs = inputs + self.weights[:, :inputs.shape[1], :]
        return self.dropout(inputs)

    def bert_init(self, inputs_size, device, dtype):
        return torch.normal(
            0, 0.02, size=[1, inputs_size[0], inputs_size[1]],
            dtype=dtype
        ).to(device)

    def original_init(self, inputs_size, device, dtype):
        max_len, num_hiddens = inputs_size
        weights = torch.zeros((1, *inputs_size))
        X = (torch.arange(max_len, dtype=dtype).reshape(-1, 1) /  # i
             torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=dtype) / num_hiddens))
        weights[:, :, 0::2] = torch.sin(X)  # 这里的切片方法[start: end: step]
        weights[:, :, 1::2] = torch.cos(X)
        return weights.to(device)

#
# X = torch.randn([4, 128, 512])
# model = AddPositionEmbs('original', X.shape[1:])
# X = model(X)
# print(X.shape)
