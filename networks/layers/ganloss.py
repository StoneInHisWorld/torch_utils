import torch
from torch import nn


class GANLoss(nn.Module):
    """定义不同的GAN目标函数

    GANLoss类将构造目标标签张量的行为进行了抽象，目标标签张量与输入张量同形。
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0, **kwargs):
        """ 初始化GANLoss类

        :param gan_mode: GAN目标函数的类型，目前支持vanilla、lsgan、wgangp.
        :param target_real_label: 真实图片标签
        :param target_fake_label: 虚假图片标签

        注：请不要使用`sigmoid()`作为分辨器的最后一层，LSGAN不允许使用`sigmoid()`。
        vanilla GANs 会使用BCEWithLogitsLoss来处理。
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss(**kwargs)
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(**kwargs)
        elif gan_mode in ['wgangp']:
            self.loss = None
            self.size_averaged = kwargs['size_averaged'] if 'size_averaged' in kwargs.keys() else True
        else:
            raise NotImplementedError(f'不支持的GAN模式{gan_mode}！' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """构造与输入同形的标签张量。

        :param prediction: 通常是分辨器的预测值。
        :param target_is_real: 真实标签为真实图片或虚假图片
        :return 含有真实标注的标签张量，且与输入同形。
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """通过分辨器输出以及真实标签计算损失值

        :param prediction: 通常是分辨器的预测值。
        :param target_is_real: 真实标签为真实图片或虚假图片
        :return 计算所得损失值
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            # if target_is_real:
            #     loss = -prediction.mean()
            # else:
            #     loss = prediction.mean()
            if target_is_real:
                loss = -prediction
            else:
                loss = prediction
            if self.size_averaged:
                loss = loss.mean()
        return loss
