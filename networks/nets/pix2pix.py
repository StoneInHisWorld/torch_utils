from collections import OrderedDict

import torch
import inspect

import utils.func.torch_tools as ttools
from networks.basic_nn import BasicNN
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
        :param g_args: 生成器位置参数
        :param g_kwargs: 生成器关键词参数
        :param d_args: 分辨器位置参数
        :param d_kwargs: 分辨器关键词参数
        :param direction: 方向，'AtoB'意为从特征集预测到标签集，'BtoA'意为从标签集预测到特征集
        :param kwargs: BasicNN关键词参数
        """
        _construction_variables = locals()
        _parameters = inspect.signature(self.__init__).parameters
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

        self._construction_variables = _construction_variables
        self._construction_parameters = _parameters

    def forward(self, input):
        """前向传播
        计算生成器的预测值。
        :param input: 输入特征批
        :return: 生成器预测图片批
        """
        return self.netG(input)

    # def __backward_G(self, X, y, pred, backward=True):
    #     """计算生成器的GAN损失值和L1损失值"""
    #     ls_es = self._ls_fn_s[0](X, y, pred)
    #     if backward:
    #         try:
    #             ls_es.backward()
    #             return ls_es.item()
    #         except:
    #             ls_es[0].backward()
    #     return [ls.item() for ls in ls_es]
    #
    # def __backward_D(self, X, y, pred, backward=True):
    #     """计算分辨器的GANLoss"""
    #     ls_es = self._ls_fn_s[1](X, y, pred)
    #     if backward:
    #         try:
    #             ls_es.backward()
    #             return ls_es.item()
    #         except:
    #             ls_es[0].backward()
    #     return [ls.item() for ls in ls_es]

    def forward_backward(self, X, y, backward=True):
        # 前向传播
        AtoB = self.direction == 'AtoB'
        X, y = [X, y] if AtoB else [y, X]
        pred = self(X)
        G_lsfn, D_lsfn = self._ls_fn_s
        if backward:
            # 需要进行反向传播
            # 取出优化器和损失函数
            optimizer_G, optimizer_D = self._optimizer_s
            self.netD.requires_grad_(True)
            optimizer_D.zero_grad()  # set D's gradients to zero
            loss_D = D_lsfn(X, y, pred)
            if isinstance(loss_D, tuple):
                loss_D[0].backward()
            else:
                loss_D.backward()
            optimizer_D.step()  # update D's weights
            # 更新生成器
            self.netD.requires_grad_(False)
            optimizer_G.zero_grad()  # set G's gradients to zero
            loss_G = G_lsfn(X, y, pred)
            if isinstance(loss_G, tuple):
                loss_G[0].backward()
            else:
                loss_G.backward()
            optimizer_G.step()  # update G's weights
        else:
            with torch.no_grad():
                loss_D = ()  # 计算分辨器的损失
                loss_G = G_lsfn(X, y, pred)  # 计算生成器的损失
        return pred, (*loss_G, *loss_D)

    def _get_optimizer(self, *args):
        """获取pix2pix的优化器。
        目前只允许给netG，netD分配优化器，且每个网络只能分配一个。
        :param args: 各个优化器的类型以及关键字参数，如果指定数目少于2个，则会用论文中的参数填充。
        :return: 优化器序列，学习率名称序列
        """
        # 如果指定参数少于2个，则使用默认值填充至2个
        if len(args) < 2:
            # 如果优化器类型指定太少，则使用默认值补充
            args = (*args, *[('adam', {}) for _ in range(2 - len(args))])
        # 检查优化器参数
        optimizers = []
        for (type_s, kwargs), net in zip(args, [self.netG, self.netD]):
            if 'lr' not in kwargs.keys():
                kwargs['lr'] = 0.2
            if 'betas' not in kwargs.keys():
                kwargs['betas'] = (0.1, 0.999)
            optimizers.append(ttools.get_optimizer(net, type_s, **kwargs))
        return optimizers, ['G_lrs', 'D_lrs']

    def _get_lr_scheduler(self, *args):
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
        scheduler_s = []
        for (ss, kwargs), optimizer in zip(args, self._optimizer_s):
            if ss == 'linear':
                epoch_count = 1 if 'epoch_count' not in kwargs.keys() else kwargs['epoch_count']
                n_epochs_decay = 100 if 'n_epochs_decay' not in kwargs.keys() else kwargs['n_epochs_decay']
                n_epochs = 100 if 'epoch_count' not in kwargs.keys() else kwargs['n_epochs']

                def lambda_rule(epoch):
                    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
                    return lr_l

                ss = 'lambda'
                kwargs['lr_lambda'] = lambda_rule
            elif ss == 'step':
                step_size = 50 if 'step_size' not in kwargs.keys() else kwargs['step_size']
                gamma = 0.1 if 'gamma' not in kwargs.keys() else kwargs['gamma']
                kwargs.update({'step_size': step_size, 'gamma': gamma})
            elif ss == 'plateau':
                mode = 'min' if 'mode' not in kwargs.keys() else kwargs['mode']
                factor = 0.2 if 'factor' not in kwargs.keys() else kwargs['factor']
                threshold = 0.01 if 'threshold' not in kwargs.keys() else kwargs['threshold']
                patience = 5 if 'patience' not in kwargs.keys() else kwargs['patience']
                kwargs.update({'mode': mode, 'factor': factor, 'threshold': threshold, 'patience': patience})
            elif ss == 'cosine':
                T_max = 100 if 'T_max' not in kwargs.keys() else kwargs['T_max']
                eta_min = 0 if 'eta_min' not in kwargs.keys() else kwargs['eta_min']
                kwargs.update({'T_max': T_max, 'eta_min': eta_min})
            scheduler_s.append(ttools.get_lr_scheduler(optimizer, ss, **kwargs))
        return scheduler_s

    def _get_ls_fn(self, *args):
        if len(args) == 0:
            args = ('cGAN', {})
        elif len(args) > 1:
            raise NotImplementedError(f'Pix2Pix暂不支持指定多种损失函数')
        ls_fns, loss_names, test_ls_names = [], [], []
        supported = ['pcc', 'cGAN']
        for (ss, kwargs) in args:
            if ss == 'cGAN':
                # # 处理关键词参数
                # try:
                #     kwargs = args[0]
                #     gan_mode = kwargs.pop('gan_mode', 'lsgan')
                #     reduced_form = kwargs.pop('reduced_form', False)
                #     size_averaged = kwargs.pop('size_averaged', True)
                #     lambda_l1 = kwargs.pop('lambda_l1', 100.)
                # except IndexError:
                #     # 如果没有传递参数
                #     gan_mode = 'lsgan'
                #     reduced_form = False
                #     lambda_l1 = 100.
                #     kwargs = {}
                #     size_averaged = True
                # # 定义损失函数
                # # criterionGAN = GANLoss(gan_mode, **kwargs).to(self.device)
                # criterionGAN = ttools.get_ls_fn('gan', gan_mode=gan_mode, device=self.device, **kwargs)
                # criterionL1 = ttools.get_ls_fn('l1', **kwargs)
                # self.criterionGAN = lambda pred, target_is_real: criterionGAN(pred, target_is_real)
                # # self.criterionL1 = lambda X, y: torch.nn.L1Loss(**kwargs)(X, y) * lambda_l1
                # self.criterionL1 = lambda X, y: criterionL1(X, y) * lambda_l1
                #
                # def G_ls_fn(X, y, pred):
                #     # 首先，G(A)需要骗过分辨器
                #     fake_AB = torch.cat((X, pred), 1)
                #     pred_fake = self.netD(fake_AB)
                #     gan_ls = self.criterionGAN(pred_fake, True)
                #     l1_ls = self.criterionL1(pred, y)
                #     if not size_averaged:
                #         gan_ls = gan_ls.mean(dim=list(range(len(pred_fake.shape)))[1:])
                #         l1_ls = l1_ls.mean(dim=list(range(len(pred.shape)))[1:])
                #     if reduced_form:
                #         return gan_ls + l1_ls
                #     else:
                #         return gan_ls + l1_ls, gan_ls, l1_ls
                #
                # def D_ls_fn(X, y, pred):
                #     fake_AB = torch.cat((X, pred), 1)
                #     pred_fake = self.netD(fake_AB.detach())
                #     fake_ls = self.criterionGAN(pred_fake, False)
                #     # 真实值
                #     real_AB = torch.cat((X, y), 1)
                #     pred_real = self.netD(real_AB)
                #     real_ls = self.criterionGAN(pred_real, True)
                #     # 组合损失值并计算梯度，0.5是两个预测值的权重
                #     ls = (fake_ls + real_ls) * 0.5
                #     if reduced_form:
                #         return ls
                #     else:
                #         return ls, real_ls, fake_ls
                #
                # ls_fn = D_ls_fn, G_ls_fn
                # loss_names = ['G_LS', 'D_LS'] if reduced_form else \
                #     ['G_LS', 'G_GAN', 'G_L1', 'D_LS', 'D_real', 'D_fake']
                # test_ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']
                ls_fn, ls_names, tls_names = self.cGAN_ls_fn(**kwargs)
            elif ss == 'pcc':
                ls_fn, ls_names, tls_names = self.pcc_ls_fn(**kwargs)
            else:
                raise NotImplementedError(f'Pix2Pix暂不支持本损失函数模式{ss}，目前支持{supported}')
            ls_fns += list(ls_fn)
            # 指定输出的训练损失类型
            loss_names += ls_names
            test_ls_names += tls_names
        return ls_fns, loss_names, test_ls_names

    def cGAN_ls_fn(self, **kwargs):
        # 处理关键词参数
        try:
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
        # GAN不接受size_averaged参数
        if not size_averaged and 'reduction' not in kwargs.keys():
            kwargs['reduction'] = 'none'
        # criterionGAN = GANLoss(gan_mode, **kwargs).to(self.device)
        criterionGAN = ttools.get_ls_fn('gan', gan_mode=gan_mode, device=self.device, **kwargs)
        criterionL1 = ttools.get_ls_fn('l1', **kwargs)
        self.criterionGAN = lambda pred, target_is_real: criterionGAN(pred, target_is_real)
        self.criterionL1 = lambda X, y: criterionL1(X, y) * lambda_l1

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

        ls_fn = G_ls_fn, D_ls_fn
        loss_names = ['G_LS', 'D_LS'] if reduced_form else \
            ['G_LS', 'G_GAN', 'G_L1', 'D_LS', 'D_real', 'D_fake']
        test_ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']
        return ls_fn, loss_names, test_ls_names

    def pcc_ls_fn(self, **kwargs):
        # 处理关键词参数
        try:
            # kwargs = args[0]
            gan_mode = kwargs.pop('gan_mode', 'lsgan')
            reduced_form = kwargs.pop('reduced_form', False)
            size_averaged = kwargs.pop('size_averaged', True)
            lambda_PCC = kwargs.pop('lambda_pcc', 1.)
        except IndexError:
            # 如果没有传递参数
            gan_mode = 'lsgan'
            reduced_form = False
            lambda_PCC = 1.
            kwargs = {}
            size_averaged = True
        # GAN、PCC不接受size_averaged参数
        if not size_averaged and 'reduction' not in kwargs.keys():
            kwargs['reduction'] = 'none'
        criterionGAN = ttools.get_ls_fn('gan', gan_mode=gan_mode, device=self.device, **kwargs)
        criterionPCC = ttools.get_ls_fn('pcc', **kwargs)
        self.criterionGAN = lambda pred, target_is_real: criterionGAN(pred, target_is_real)
        self.criterionPCC = lambda X, y: criterionPCC(X, y) * lambda_PCC

        def G_ls_fn(X, y, pred):
            # 首先，G(A)需要骗过分辨器
            fake_AB = torch.cat((X, pred), 1)
            pred_fake = self.netD(fake_AB)
            gan_ls = self.criterionGAN(pred_fake, True)
            pcc_ls = self.criterionPCC(pred, y)
            if not size_averaged:
                gan_ls = gan_ls.mean(dim=list(range(len(pred_fake.shape)))[1:])
                # pcc_ls = pcc_ls.mean(dim=list(range(len(pred.shape)))[1:])
            if reduced_form:
                return gan_ls + pcc_ls
            else:
                return gan_ls + pcc_ls, gan_ls, pcc_ls

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

        ls_fn = G_ls_fn, D_ls_fn
        loss_names = ['G_LS', 'D_LS'] if reduced_form else \
            ['G_LS', 'G_GAN', 'G_PCC', 'D_LS', 'D_real', 'D_fake']
        test_ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_PCC']
        return ls_fn, loss_names, test_ls_names

    @property
    def required_shape(self):
        return (256, 256)