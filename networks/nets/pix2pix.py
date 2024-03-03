from collections import OrderedDict

import torch
from torch import nn as nn
from torch.nn import Module
from torch.optim import lr_scheduler
from tqdm import tqdm

from networks.basic_nn import BasicNN
from networks.layers.ganloss import GANLoss
from networks.nets.pix2pix_d import Pix2Pix_D
from networks.nets.pix2pix_g import Pix2Pix_G


class Pix2Pix(BasicNN):

    def __init__(self,
                 g_args, g_kwargs,
                 d_args, d_kwargs,
                 gan_mode='lsgan', lr=0.2, beta1=0.1,
                 direction='AtoB', isTrain=True,
                 *args: Module, **kwargs):
        """
        实现pix2pix模型，通过给定数据对学习输入图片到输出图片的映射。

        pix2pix不使用图片缓存。

        目标函数为 :math:`GAN Loss + lambda_{L1} * ||G(A)-B||_1`
        :param g_args:
        :param g_kwargs:
        :param d_args:
        :param d_kwargs:
        :param gan_mode:
        :param lr:
        :param beta1:
        :param direction:
        :param args:
        :param kwargs:
        """
        super(Pix2Pix, self).__init__(*args, **kwargs)
        # 指定输出的训练损失类型
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # 指定保存或展示的图片
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.direction = direction
        # 指定保存的模型
        if isTrain:
            self.model_names = ['G', 'D']
        else:  # 测试时只加载生成器
            self.model_names = ['G']
        # 定义生成器和分辨器
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = Pix2Pix_G(*g_args, **g_kwargs)

        if isTrain:
            # 定义一个分辨器
            # conditional GANs需要输入和输出图片，因此分辨器的通道数为input_nc + output_nc
            # self.netD = Pix2Pix_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = Pix2Pix_D(*d_args, **d_kwargs)

        if isTrain:
            # 定义损失函数
            self.criterionGAN = GANLoss(gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # 初始化优化器
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=lr, betas=(beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=lr, betas=(beta1, 0.999)
            )
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def forward(self, input):
        input_A = input[0]
        input_B = input[1]
        AtoB = self.direction == 'AtoB'
        self.real_A = input_A if AtoB else input_B
        self.real_B = input_B if AtoB else input_A
        self.fake_B = self.netG(self.real_A)

    def backward_G(self, lambda_l1=100.0):
        """计算生成器的GAN损失值和L1损失值"""
        # 首先，G(A)需要骗过分辨器
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # 接下来，计算G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_l1
        # 组合损失值并计算梯度
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_D(self):
        """计算分辨器的GANLoss"""
        # 造假。将fake_B解绑以停止向生成器的反向传播
        # 使用conditional GAN，需要向分辨器投入输入和输出
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # 真实值
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # 组合损失值并计算梯度
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize_parameters(self):
        # self.forward()                   # 计算虚假图片：G(A)
        # 更新分辨器
        # self.netD.requires_grad_(True)  # TODO: 请检查与下个语句是否等效？
        self.set_requires_grad(self.netD, True)  # 开启分辨器的反向传播
        self.optimizer_D.zero_grad()     # 清零分辨器的梯度
        self.backward_D()                # 计算分辨器的梯度
        self.optimizer_D.step()          # 更新分辨器的权重
        # 更新生成器
        # self.netD.requires_grad_(False)  # TODO: 请检查与下个语句是否等效？
        self.set_requires_grad(self.netD, False)  # 优化生成器时，分辨器不需要计算梯度
        self.optimizer_G.zero_grad()        # 清零生成器的梯度
        self.backward_G()                   # 计算生成器的梯度
        self.optimizer_G.step()             # 更新生成器的权重

    def train_(self,
               data_iter, optimizer, acc_fn,
               n_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
               lr_scheduler=None, valid_iter=None,
               k=1, n_workers=1, hook=None,
               lr_scheduler_args=()
               ):
        self.schedulers = [
            Pix2Pix.get_scheduler(optimizer, lr_scheduler,
                                  *lr_scheduler_args)
            for optimizer in self.optimizers
        ]
        with tqdm(total=len(data_iter) * n_epochs, unit='批', position=0,
                  desc=f'训练中...', mininterval=1) as pbar:
            for epoch in range(n_epochs):
                pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中...')
                for data in data_iter:
                    self(data)
                    self.optimize_parameters()
                    ls = self.get_current_losses()

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    @staticmethod
    def get_scheduler(optimizer, lr_policy='linear',
                      *args):
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
        if lr_policy == 'linear':
            epoch_count, n_epochs, n_epochs_decay = args

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            lr_decay_iters, = args
            scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
        elif lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            n_epochs, = args
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
        return scheduler

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @property
    def required_shape(self):
        return (256, 256)
