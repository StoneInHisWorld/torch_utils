from collections import OrderedDict

import torch
from torch.optim import lr_scheduler

import utils.func.torch_tools as ttools
from networks.basic_nn import BasicNN
from networks.layers.ganloss import GANLoss
from networks.nets.pix2pix_d import Pix2Pix_D
from networks.nets.pix2pix_g import Pix2Pix_G


class Pix2Pix(BasicNN):

    def __init__(self,
                 g_args, g_kwargs,
                 d_args, d_kwargs,
                 direction='AtoB', isTrain=True,
                 **kwargs):
        """实现pix2pix模型，通过给定数据对学习输入图片到输出图片的映射。

        pix2pix不使用图片缓存。

        生成器目标函数为 :math:`\ell_G = G_GAN + \lambda_{L1}||G(A)-B||_1`

        分辨器目标函数为 :math:`\ell_D = 0.5 D_read + 0.5 D_fake`

        训练过程中输出四个目标值，分别为G_GAN、G_L1、D_real、D_fake：

        - :math:`G_GAN`：生成器的GAN损失
        - :math:`G_{L1}`：生成器的L1损失
        - :math:`D_real`：分辨器分辨真图为真的概率，其输入为真实标签。
        - :math:`D_fake`：分辨器分辨合成图为假的概率，其输入为生成器合成图。
        :param g_args:
        :param g_kwargs:
        :param d_args:
        :param d_kwargs:
        :param direction:
        :param kwargs:
        """
        self.direction = direction
        netG = Pix2Pix_G(*g_args, **g_kwargs)

        if isTrain:
            # 定义一个分辨器
            # conditional GANs需要输入和输出图片，因此分辨器的通道数为input_nc + output_nc
            netD = Pix2Pix_D(*d_args, **d_kwargs)
            super(Pix2Pix, self).__init__(OrderedDict([
                ('netG', netG), ('netD', netD)
            ]), **kwargs)
        else:
            super(Pix2Pix, self).__init__(netG, **kwargs)

    def forward(self, input):
        """前向传播
        计算生成器的预测值。
        :param input: 输入特征批
        :return: 生成器预测图片批
        """
        return self.netG(input)

    def __backward_G(self, X, y, pred, backward=True):
        """计算生成器的GAN损失值和L1损失值"""
        ls_es = self._ls_fn_s[0](X, y, pred)
        if backward:
            try:
                ls_es.backward()
                return ls_es.item()
            except:
                ls_es[0].backward()
        return [ls.item() for ls in ls_es]

    def __backward_D(self, X, y, pred, backward=True):
        """计算分辨器的GANLoss"""
        ls_es = self._ls_fn_s[1](X, y, pred)
        if backward:
            try:
                ls_es.backward()
                return ls_es.item()
            except:
                ls_es[0].backward()
        return [ls.item() for ls in ls_es]

    def forward_backward(self, X, y, backward=True):
        # 前向传播
        AtoB = self.direction == 'AtoB'
        X, y = [X, y] if AtoB else [y, X]
        pred = self(X)
        if backward:
            # 需要进行反向传播
            loss_es = []
            for requires_grad, ls_fn, optim in zip(
                    [True, False], self._ls_fn_s,
                    [self.optimizer_D, self.optimizer_G]
            ):
                # 依次进行分辨器和生成器的梯度计算、损失值计算以及参数更新
                self.netD.requires_grad_(requires_grad)
                optim.zero_grad()
                ls_es = ls_fn(X, y, pred)
                self._backward_impl(*ls_es)
                optim.step()
                loss_es.append(ls_es)
            loss_D, loss_G = loss_es
        else:
            with torch.no_grad():
                loss_D = ()  # 计算分辨器的损失
                loss_G = self._ls_fn_s[1](X, y, pred)  # 计算生成器的损失
        return pred, (*loss_G, *loss_D)

    def _get_optimizer(self, optim_str_s=None, *args):
        # 检查优化器类型
        if optim_str_s is None:
            optim_str_s = []
        if len(optim_str_s) < 2:
            # 如果优化器类型指定太少，则使用默认值补充
            optim_str_s = (*optim_str_s, *['adam' for _ in range(2 - len(optim_str_s))])
        optim_str_g, optim_str_d = optim_str_s
        # 检查优化器参数
        try:
            g_kwargs, d_kwargs = args
            for kwargs in [g_kwargs, d_kwargs]:
                if 'lr' not in kwargs.keys():
                    kwargs['lr'] = 0.2
                if 'betas' not in kwargs.keys():
                    kwargs['betas'] = (0.1, 0.999)
        except TypeError:
            raise TypeError(f'{self.__name__}需要在args参数下同时为生成器和分辨器的优化器指定参数。')
        except ValueError:
            # 设置默认参数
            kwargs = {'lr': 0.2, 'betas': (0.1, 0.999)}
            if len(args) == 0:
                g_kwargs, d_kwargs = kwargs, kwargs
            elif len(args) == 1:
                g_kwargs, d_kwargs = args[0], kwargs
            else:
                raise ValueError(f'{self.__name__}在args参数下只需要为生成器和分辨器的优化器指定参数。')
        self.optimizer_G = ttools.get_optimizer(
            self.netG, optim_str_g, **g_kwargs
        )
        self.optimizer_D = ttools.get_optimizer(
            self.netD, optim_str_d, **d_kwargs
        )
        self.lr_names = ['G_lrs', 'D_lrs']
        return [self.optimizer_G, self.optimizer_D]

    def _get_lr_scheduler(self, scheduler_str_s=None, *args):
        """Return a learning rate scheduler

        Parameters:
            optimizer          -- the optimizer of the network
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                                  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

        For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
        and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        """
        if scheduler_str_s is None:
            scheduler_str_s = []
        if len(args) < len(scheduler_str_s):
            # 如果优化器类型指定太少，则使用默认值补充
            args = (*args, *[{} for _ in range(len(scheduler_str_s) - len(args))])
        scheduler_s = []
        for ss, optimizer, kwargs in zip(scheduler_str_s, self._optimizer_s, args):
            if ss == 'linear':
                epoch_count = 1 if 'epoch_count' not in kwargs.keys() else kwargs['epoch_count']
                n_epochs_decay = 100 if 'n_epochs_decay' not in kwargs.keys() else kwargs['n_epochs_decay']
                n_epochs = 100 if 'epoch_count' not in kwargs.keys() else kwargs['n_epochs']

                def lambda_rule(epoch):
                    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
                    return lr_l

                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            elif ss == 'step':
                step_size = 50 if 'step_size' not in kwargs.keys() else kwargs['step_size']
                gamma = 0.1 if 'gamma' not in kwargs.keys() else kwargs['gamma']
                kwargs.update({'step_size': step_size, 'gamma': gamma})
                scheduler = lr_scheduler.StepLR(optimizer, **kwargs)
            elif ss == 'plateau':
                mode = 'min' if 'mode' not in kwargs.keys() else kwargs['mode']
                factor = 0.2 if 'factor' not in kwargs.keys() else kwargs['factor']
                threshold = 0.01 if 'threshold' not in kwargs.keys() else kwargs['threshold']
                patience = 5 if 'patience' not in kwargs.keys() else kwargs['patience']
                kwargs.update({'mode': mode, 'factor': factor, 'threshold': threshold, 'patience': patience})
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            elif ss == 'cosine':
                T_max = 100 if 'T_max' not in kwargs.keys() else kwargs['T_max']
                eta_min = 0 if 'eta_min' not in kwargs.keys() else kwargs['eta_min']
                kwargs.update({'T_max': T_max, 'eta_min': eta_min})
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
            else:
                return NotImplementedError(f'不支持的学习率变化策略{scheduler_str_s}')
            scheduler_s.append(scheduler)
        return scheduler_s

    def _get_ls_fn(self, ls_fn='cGAN', *args):
        supported = ['cGAN']
        if ls_fn == 'cGAN':
            # 处理关键词参数
            try:
                kwargs = args[0]
                gan_mode = kwargs.pop('gan_mode', 'lsgan')
                reduced_form = kwargs.pop('reduced_form', False)
                size_averaged = kwargs.pop('size_averaged', True)
                lambda_l1 = kwargs.pop('lambda_l1', 100.)
            except IndexError:
                # 如果没有传递参数
                gan_mode = 'lsgan'
                reduced_form = False
                lambda_l1 = 100.
                kwargs = {}
                size_averaged = True
            # 定义损失函数
            criterionGAN = GANLoss(gan_mode, **kwargs).to(self.device)
            self.criterionGAN = lambda pred, target_is_real: criterionGAN(pred, target_is_real)
            self.criterionL1 = lambda X, y: torch.nn.L1Loss(**kwargs)(X, y) * lambda_l1

            def G_ls_fn(X, y, pred):
                # 首先，G(A)需要骗过分辨器
                fake_AB = torch.cat((X, pred), 1)
                pred_fake = self.netD(fake_AB)
                gan_ls = self.criterionGAN(pred_fake, True)
                l1_ls = self.criterionL1(pred, y)
                if not size_averaged:
                    gan_ls = gan_ls.mean(dim=list(range(len(pred_fake.shape)))[1:])
                    l1_ls = l1_ls.mean(dim=list(range(len(pred.shape)))[1:])
                if reduced_form:
                    return gan_ls + l1_ls
                else:
                    return gan_ls + l1_ls, gan_ls, l1_ls

            def D_ls_fn(X, y, pred):
                fake_AB = torch.cat((X, pred), 1)
                pred_fake = self.netD(fake_AB.detach())
                fake_ls = self.criterionGAN(pred_fake, False)
                # 真实值
                real_AB = torch.cat((X, y), 1)
                pred_real = self.netD(real_AB)
                real_ls = self.criterionGAN(pred_real, True)
                # 组合损失值并计算梯度，0.5是两个预测值的权重
                ls = (fake_ls + real_ls) * 0.5
                if reduced_form:
                    return ls
                else:
                    return ls, real_ls, fake_ls

            ls_fn = D_ls_fn, G_ls_fn
            loss_names = ['G_LS', 'D_LS'] if reduced_form else \
                ['G_LS', 'G_GAN', 'G_L1', 'D_LS', 'D_real', 'D_fake']
            test_ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']
        else:
            raise NotImplementedError(f'Pix2Pix暂不支持本损失函数模式{ls_fn}，目前支持{supported}')
        # 指定输出的训练损失类型
        self.loss_names, self.test_ls_names = loss_names, test_ls_names
        return ls_fn

    @property
    def required_shape(self):
        return (256, 256)