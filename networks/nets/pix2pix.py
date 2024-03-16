from collections import OrderedDict
from typing import Callable, Tuple, List

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

import utils.func.torch_tools as ttools
from networks.basic_nn import BasicNN
from networks.layers.ganloss import GANLoss
from networks.nets.pix2pix_d import Pix2Pix_D
from networks.nets.pix2pix_g import Pix2Pix_G
from utils.accumulator import Accumulator
from utils.history import History


class Pix2Pix(BasicNN):

    def __init__(self,
                 g_args, g_kwargs,
                 d_args, d_kwargs,
                 gan_mode='lsgan', direction='AtoB', lambda_l1=100.,
                 isTrain=True,
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
        :param gan_mode:
        :param direction:
        :param kwargs:
        """
        # 指定保存或展示的图片
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.direction = direction
        # # 指定保存的模型
        # if isTrain:
        #     self.model_names = ['G', 'D']
        # else:  # 测试时只加载生成器
        #     self.model_names = ['G']
        # 定义生成器和分辨器
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        netG = Pix2Pix_G(*g_args, **g_kwargs)

        if isTrain:
            # 定义一个分辨器
            # conditional GANs需要输入和输出图片，因此分辨器的通道数为input_nc + output_nc
            # self.netD = Pix2Pix_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                       opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            netD = Pix2Pix_D(*d_args, **d_kwargs)
            super(Pix2Pix, self).__init__(OrderedDict([
                ('netG', netG), ('netD', netD)
            ]), **kwargs)
            # self.netD = lambda input: self[0](input)
            # self.netG = lambda input: self[1](input)
        else:
            super(Pix2Pix, self).__init__(netG, **kwargs)
            # self.netG = self[0]

        # if isTrain:
        # # 定义损失函数
        # self.criterionGAN = GANLoss(gan_mode).to(self.device)
        # self.criterionL1 = lambda X, y: torch.nn.L1Loss()(X, y) * lambda_l1
        # # 初始化优化器
        # self.optimizer_G = torch.optim.Adam(
        #     self.netG.parameters(), lr=lr, betas=(beta1, 0.999)
        # )
        # self.optimizer_D = torch.optim.Adam(
        #     self.netD.parameters(), lr=lr, betas=(beta1, 0.999)
        # )
        # self.optimizers = []
        # self.optimizers.append(self.optimizer_G)
        # self.optimizers.append(self.optimizer_D)

        # self.isTrain = isTrain

    def forward(self, input):
        # input_A = input[0]
        # input_B = input[1]
        # AtoB = self.direction == 'AtoB'
        # self.real_A = input_A if AtoB else input_B
        # self.real_B = input_B if AtoB else input_A
        # self.fake_B = self.netG(self.real_A)
        # return self.fake_B
        return self.netG(input)

    # def backward_G(self, lambda_l1=100.0):
    #     """计算生成器的GAN损失值和L1损失值"""
    #     # 首先，G(A)需要骗过分辨器
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake = self.netD(fake_AB)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)
    #     # 接下来，计算G(A) = B
    #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_l1
    #     # 组合损失值并计算梯度
    #     self.loss_G = self.loss_G_GAN + self.loss_G_L1
    #     self.loss_G.backward()
    #
    # def backward_D(self):
    #     """计算分辨器的GANLoss"""
    #     # 造假。将fake_B解绑以停止向生成器的反向传播
    #     # 使用conditional GAN，需要向分辨          器投入输入和输出
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # 真实值
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # 组合损失值并计算梯度，0.5是两个预测值的权重
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()

    # def optimize_parameters(self):
    #     # self.forward()                   # 计算虚假图片：G(A)
    #     # 更新分辨器
    #     self.netD.requires_grad_(True)  # TODO: 请检查与下个语句是否等效？
    #     # self.set_requires_grad(self.netD, True)  # 开启分辨器的反向传播
    #     self.optimizer_D.zero_grad()     # 清零分辨器的梯度
    #     self.backward_D()                # 计算分辨器的梯度
    #     self.optimizer_D.step()          # 更新分辨器的权重
    #     # 更新生成器
    #     self.netD.requires_grad_(False)  # TODO: 请检查与下个语句是否等效？
    #     # self.set_requires_grad(self.netD, False)  # 优化生成器时，分辨器不需要计算梯度
    #     self.optimizer_G.zero_grad()        # 清零生成器的梯度
    #     self.backward_G()                   # 计算生成器的梯度
    #     self.optimizer_G.step()             # 更新生成器的权重

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
        # """计算生成器的GAN损失值和L1损失值"""
        # # 首先，G(A)需要骗过分辨器
        # fake_AB = torch.cat((X, pred), 1)
        # pred_fake = self.netD(fake_AB)
        # gan_ls = self.criterionGAN(pred_fake, True)
        # # 接下来，计算G(A) = B
        # l1_ls = self.criterionL1(pred, y)
        # # 组合损失值并计算梯度
        # ls = gan_ls + l1_ls
        # if backward:
        #     ls.backward()
        # return ls.item(), gan_ls.item(), l1_ls.item()

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

    #   """计算分辨器的GANLoss"""
    # # 造假。将fake_B解绑以停止向生成器的反向传播
    # # 使用conditional GAN，需要向分辨器投入输入和输出
    # fake_AB = torch.cat((X, pred), 1)
    # pred_fake = self.netD(fake_AB.detach())
    # fake_ls = self.criterionGAN(pred_fake, False)
    # # 真实值
    # real_AB = torch.cat((X, y), 1)
    # pred_real = self.netD(real_AB)
    # real_ls = self.criterionGAN(pred_real, True)
    # # 组合损失值并计算梯度，0.5是两个预测值的权重
    # ls = (fake_ls + real_ls) * 0.5
    # if backward:
    #     ls.backward()
    # return ls.item(), real_ls.item(), fake_ls.item()

    # def backward(self, X, y):
    #     # # 更新分辨器
    #     # self.netD.requires_grad_(True)  # TODO: 请检查与下个语句是否等效？
    #     # self.optimizer_D.zero_grad()     # 清零分辨器的梯度
    #     # loss_D = backward_D()                # 计算分辨器的梯度
    #     # self.optimizer_D.step()          # 更新分辨器的权重
    #     # # 更新生成器
    #     # self.netD.requires_grad_(False)  # TODO: 请检查与下个语句是否等效？
    #     # self.optimizer_G.zero_grad()        # 清零生成器的梯度
    #     # loss_G = backward_G()                   # 计算生成器的梯度
    #     # self.optimizer_G.step()             # 更新生成器的权重
    #     # return loss_G, loss_D
    #     # 前向传播
    #     AtoB = self.direction == 'AtoB'
    #     X, y = [X, y] if AtoB else [y, X]
    #     pred = self(X)
    #     # 更新分辨器
    #     self.netD.requires_grad_(True)
    #     self.optimizer_D.zero_grad()  # 清零分辨器的梯度
    #     loss_D = backward_D(X, y, pred)  # 计算分辨器的梯度
    #     self.optimizer_D.step()  # 更新分辨器的权重
    #     # 更新生成器
    #     self.netD.requires_grad_(False)
    #     self.optimizer_G.zero_grad()  # 清零生成器的梯度
    #     loss_G = backward_G(X, y, pred)  # 计算生成器的梯度
    #     self.optimizer_G.step()  # 更新生成器的权重
    #     return pred, (*loss_G, *loss_D)

    def forward_backward(self, X, y, backward=True):
        # 前向传播
        AtoB = self.direction == 'AtoB'
        X, y = [X, y] if AtoB else [y, X]
        pred = self(X)
        if backward:
            # # 更新分辨器
            # self.netD.requires_grad_(True)
            # self.optimizer_D.zero_grad()  # 清零分辨器的梯度
            loss_es = []
            for requires_grad, ls_fn, optim in zip(
                    [True, False], self._ls_fn_s,
                    [self.optimizer_D, self.optimizer_G]
            ):
                self.netD.requires_grad_(requires_grad)
                optim.zero_grad()
                ls_es = ls_fn(X, y, pred)
                self._backward_impl(*ls_es)
                optim.step()
                loss_es.append(ls_es)
            loss_D, loss_G = loss_es
            # loss_D = self.__backward_D(X, y, pred)  # 计算分辨器的梯度
            # self.optimizer_D.step()  # 更新分辨器的权重
            # # 更新生成器
            # self.netD.requires_grad_(False)
            # self.optimizer_G.zero_grad()  # 清零生成器的梯度
            # loss_G = self.__backward_G(X, y, pred)  # 计算生成器的梯度
            # self.optimizer_G.step()  # 更新生成器的权重
        else:
            with torch.no_grad():
                # loss_D = self.__backward_D(X, y, pred, False)  # 计算分辨器的损失
                loss_D = ()  # 计算分辨器的损失
                # loss_G = self.__backward_G(X, y, pred, False)  # 计算生成器的损失
                loss_G = self._ls_fn_s[1](X, y, pred)  # 计算生成器的损失
        return pred, (*loss_G, *loss_D)
        # # 前向传播
        # AtoB = self.direction == 'AtoB'
        # X, y = [X, y] if AtoB else [y, X]
        # pred = self(X)
        # if backward:
        #     # 更新分辨器
        #     self.netD.requires_grad_(True)
        #     self.optimizer_D.zero_grad()  # 清零分辨器的梯度
        #     loss_D = self.__backward_D(X, y, pred)  # 计算分辨器的梯度
        #     self.optimizer_D.step()  # 更新分辨器的权重
        #     # 更新生成器
        #     self.netD.requires_grad_(False)
        #     self.optimizer_G.zero_grad()  # 清零生成器的梯度
        #     loss_G = self.__backward_G(X, y, pred)  # 计算生成器的梯度
        #     self.optimizer_G.step()  # 更新生成器的权重
        # else:
        #     with torch.no_grad():
        #         # loss_D = self.__backward_D(X, y, pred, False)  # 计算分辨器的损失
        #         loss_D = ()  # 计算分辨器的损失
        #         loss_G = self.__backward_G(X, y, pred, False)  # 计算生成器的损失
        # return pred, (*loss_G, *loss_D)

    def _get_optimizer(self, optim_str_s=None, *args,
                       **kwargs):
        """

        :param optim_str_s:
        :param args:
        :param kwargs:
        :return:
        """
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
        except TypeError:
            raise TypeError(f'{self.__name__}需要在args参数下同时为生成器和分辨器的优化器指定参数。')
        except ValueError:
            # TODO: 去掉args，添加默认betas值
            # 设置默认参数
            lr, betas = 0.2, (0.1, 0.999)
            if len(args) == 0:
                args = (lr,)
                kwargs = {'betas': betas}
                g_args, g_kwargs = args, kwargs
                d_args, d_kwargs = args, kwargs
            elif len(args) == 1:
                g_args, g_kwargs = args
                d_args, d_kwargs = (lr,), {'betas': betas}
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
                size_averaged = False
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

    # def train_(self,
    #            data_iter, criteria,
    #            n_epochs=10, valid_iter=None, k=1, n_workers=1, hook=None
    #            ):
    #     history = History('train_l', 'train_acc', 'lrs')
    #     with tqdm(total=len(data_iter) * n_epochs, unit='批', position=0,
    #               desc=f'训练中...', mininterval=1) as pbar:
    #         for epoch in range(n_epochs):
    #             pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中...')
    #             history.add(
    #                 ['lrs'],
    #                 [[param['lr'] for param in self.optimizer_D.param_groups + self.optimizer_G.param_groups]]
    #             )
    #             metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
    #             for data in data_iter:
    #                 self(data)
    #                 self.optimize_parameters()
    #                 ls = self.get_current_losses()
    #                 with torch.no_grad():
    #                     correct = criteria(self.fake_B, self.real_B)
    #                     num_examples = self.fake_B.shape[0]
    #                     metric.add(ls['G_GAN'] * num_examples, correct, num_examples)
    #                 pbar.update(1)
    #             # 记录训练数据
    #             history.add(
    #                 ['train_l', 'train_acc'],
    #                 [metric[0] / metric[2], metric[1] / metric[2]]
    #             )
    #         pbar.close()
    #     return history

    # def train_(self,
    #            data_iter, criterion_a,
    #            n_epochs=10, valid_iter=None, k=1, n_workers=1, hook=None
    #            ):
    #     criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
    #     # 损失项
    #     loss_names = [f'train_{item}' for item in self.loss_names]
    #     # 评价项
    #     criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
    #     # 学习率项
    #     lr_names = self.lr_names
    #     history = History(*(loss_names + criteria_names + lr_names))
    #     with tqdm(total=len(data_iter) * n_epochs, unit='批', position=0,
    #               desc=f'训练中...', mininterval=1) as pbar:
    #         for epoch in range(n_epochs):
    #             pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中...')
    #             history.add(
    #                 lr_names, [
    #                     [param['lr'] for param in optimizer.param_groups]
    #                     for optimizer in self._optimizer_s
    #                 ]
    #             )
    #             metric = Accumulator(
    #                 len(loss_names + criteria_names) + 1
    #             )  # 批次训练损失总和，准确率，样本数
    #             for X, y in data_iter:
    #                 pred, ls_es = self.forward_backward(X, y)
    #                 with torch.no_grad():
    #                     num_examples = X.shape[0]
    #                     correct_s = []
    #                     for criterion in criterion_a:
    #                         correct = criterion(pred, y)
    #                         correct_s.append(correct)
    #                     metric.add(
    #                         *[ls * num_examples for ls in ls_es],
    #                         *correct_s, num_examples
    #                     )
    #                 pbar.update(1)
    #             for scheduler in self._scheduler_s:
    #                 scheduler.step()
    #             # 记录训练数据
    #             history.add(
    #                 loss_names + criteria_names,
    #                 [metric[i] / metric[-1] for i in range(len(metric) - 1)]
    #             )
    #         pbar.close()
    #     return history

    # @torch.no_grad()
    # def test_(self, test_iter,
    #           criterion_a: Callable[[torch.Tensor, torch.Tensor], float or torch.Tensor],
    #           is_valid: bool = False, ls_fn=None, **ls_fn_kwargs
    #           ) -> dict:
    #     self.eval()
    #     # 要统计的数据种类数目
    #     criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
    #     # TODO：损失需要进行动态处理
    #     loss_names = self.loss_names if isinstance(self.loss_names, list) else [self.loss_names]
    #     metric = Accumulator(len(criterion_a) + len(loss_names) + 1)
    #     # 逐个批次计算测试数据
    #     for features, labels in test_iter:
    #         preds = self(features)
    #         fake_AB = torch.cat((features, preds), 1)
    #         pred_fake = self.netD(fake_AB)
    #         loss_G_GAN = self.criterionGAN(pred_fake, True)
    #         # 接下来，计算G(A) = B
    #         loss_G_L1 = self.criterionL1(preds, labels)
    #         # 组合损失值并计算梯度
    #         loss_G = loss_G_GAN + loss_G_L1
    #         metric.add(
    #             *[criterion(preds, labels) for criterion in criterion_a],
    #             *[ls * len(features) for ls in [loss_G_GAN, loss_G_L1, loss_G]], len(features)
    #         )
    #     # 生成测试日志
    #     log = {}
    #     prefix = 'valid_' if is_valid else 'test_'
    #     for i, computer in enumerate(criterion_a):
    #         try:
    #             log[prefix + computer.__name__] = metric[i] / metric[-1]
    #         except AttributeError:
    #             log[prefix + computer.__class__.__name__] = metric[i] / metric[-1]
    #     log[prefix + 'G_GAN'] = metric[-4] / metric[-1]
    #     log[prefix + 'G_L1'] = metric[-3] / metric[-1]
    #     log[prefix + 'G_LS'] = metric[-2] / metric[-1]
    #     # i = 0
    #     # test_log = {}
    #     # if isinstance(criterion_a, list):
    #     #     j = 0
    #     #     for j, criterion in enumerate(criterion_a):
    #     #         try:
    #     #             test_log[f'test_{criterion.__name__}'] = metric[i + j] / metric[-1]
    #     #         except AttributeError:
    #     #             test_log[f'test_{criterion.__class__.__name__}'] = metric[i + j] / metric[-1]
    #     #     i += j
    #     # else:
    #     #     try:
    #     #         test_log[f'test_{criterion_a.__name__}'] = metric[i] / metric[-1]
    #     #     except AttributeError:
    #     #         test_log[f'test_{criterion_a.__class__.__name__}'] = metric[i] / metric[-1]
    #     #     i += 1
    #     # if isinstance(loss_names, list):
    #     #     j = 0
    #     #     for j, ls_fn in enumerate(loss_names):
    #     #         try:
    #     #             test_log[f'test_{ls_fn.__name__}'] = metric[i + j] / metric[-1]
    #     #         except AttributeError:
    #     #             test_log[f'test_{ls_fn.__class__.__name__}'] = metric[i] / metric[-1]
    #     #     i += j
    #     # else:
    #     #     try:
    #     #         test_log[f'test_{loss_names.__name__}'] = metric[i] / metric[-1]
    #     #     except AttributeError:
    #     #         test_log[f'test_{loss_names.__class__.__name__}'] = metric[i] / metric[-1]
    #     #     i += 1
    #     return log

    # def get_current_losses(self):
    #     """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
    #     errors_ret = OrderedDict()
    #     for name in self.loss_names:
    #         if isinstance(name, str):
    #             errors_ret[name] = float(
    #                 getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
    #     return errors_ret

    # @staticmethod
    # def get_scheduler(optimizer, lr_policy='linear',
    #                   *args):
    #     """Return a learning rate scheduler
    #
    #     Parameters:
    #         optimizer          -- the optimizer of the network
    #         opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
    #                               opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    #
    #     For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    #     and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    #     For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    #     See https://pytorch.org/docs/stable/optim.html for more details.
    #     """
    #     if lr_policy == 'linear':
    #         epoch_count, n_epochs, n_epochs_decay = args
    #
    #         def lambda_rule(epoch):
    #             lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    #             return lr_l
    #
    #         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    #     elif lr_policy == 'step':
    #         lr_decay_iters, = args
    #         scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    #     elif lr_policy == 'plateau':
    #         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    #     elif lr_policy == 'cosine':
    #         n_epochs, = args
    #         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    #     else:
    #         return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    #     return scheduler
    #
    # def set_requires_grad(self, nets, requires_grad=False):
    #     """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    #     Parameters:
    #         nets (network list)   -- a list of networks
    #         requires_grad (bool)  -- whether the networks require gradients or not
    #     """
    #     if not isinstance(nets, list):
    #         nets = [nets]
    #     for net in nets:
    #         if net is not None:
    #             for param in net.parameters():
    #                 param.requires_grad = requires_grad

    @property
    def required_shape(self):
        return (256, 256)

    # @property
    # def loss_name(self):
    #     if self.is_train:
    #         return ['G_LS', 'D_LS'] if reduced_form else \
    #             ['G_LS', 'G_GAN', 'G_L1', 'D_LS', 'D_real', 'D_fake']
    #     else:
    #         return ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']