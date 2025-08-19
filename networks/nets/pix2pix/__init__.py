import functools

from ... import check_prepare_args

supported_ls_fns = ['pcc', 'cGAN']

import torch
import utils.func.torch_tools as ttools


def _get_ls_fn(is_train, GorD, *args):
    assert len(args) <= 1, f'Pix2Pix暂不支持指定多种损失函数'
    # 设置默认值
    if len(args) == 0:
        args = ('cGAN', {})
    ls_fn_s, ls_name_s = [], []
    for i_args in args:
        ss, kwargs = check_prepare_args(*i_args)
        # 根据ss判断损失函数类型
        if ss == 'cGAN':
            ls_fn, ls_name = cGAN_ls_fn(is_train, GorD, **kwargs)
        elif ss == 'pcc':
            ls_fn, ls_name = pcc_ls_fn(is_train, GorD, **kwargs)
        else:
            raise NotImplementedError(f'Pix2Pix_G暂不支持本损失函数模式{ss}，目前支持{supported_ls_fns}')
        ls_fn_s += ls_fn
        # 指定输出的训练损失类型
        ls_name_s += ls_name
    return ls_fn_s, ls_name_s


def l1loss(X, y, computer=ttools.get_ls_fn('l1'), lambda_l1=100):
    return computer(X, y) * lambda_l1


def cGAN_ls_fn(is_train, GorD, **kwargs):
    # 处理关键词参数
    # try:
    gan_mode = kwargs.pop('gan_mode', 'lsgan')
    reduced_form = kwargs.pop('reduced_form', False)
    size_averaged = kwargs.pop('size_averaged', True)
    lambda_l1 = kwargs.pop('lambda_l1', 100.)
    device = kwargs.pop('device', torch.device('cpu'))
    # except IndexError:
    #     # 如果没有传递参数
    #     gan_mode = 'lsgan'
    #     reduced_form = False
    #     lambda_l1 = 100.
    #     kwargs = {}
    #     size_averaged = True
    # GAN不接受size_averaged参数
    if not size_averaged and 'reduction' not in kwargs.keys():
        kwargs['reduction'] = 'none'
    criterionGAN = ttools.get_ls_fn('gan', gan_mode=gan_mode, device=device, **kwargs)
    criterionL1 = ttools.get_ls_fn('l1', **kwargs)
    # criterionGAN = lambda pred, target_is_real: criterionGAN(pred, target_is_real)
    # criterionL1 = lambda X, y: criterionL1(X, y) * lambda_l1
    criterionL1 = functools.partial(l1loss, computer=criterionL1, lambda_l1=lambda_l1)

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

    if is_train:
        ls_fn = [functools.partial(cGAN_G_ls_fn, criterionGAN=criterionGAN, criterionL1=criterionL1,
                                   size_averaged=size_averaged, reduced_form=reduced_form)] if GorD \
            else [functools.partial(D_ls_fn, criterionGAN=criterionGAN, reduced_form=reduced_form)]
        if GorD:
            ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']
        else:
            ls_names = ['D_LS'] if reduced_form else ['D_LS', 'D_real', 'D_fake']
    else:
        ls_fn = [functools.partial(cGAN_G_ls_fn, criterionGAN=criterionGAN, criterionL1=criterionL1,
                                   size_averaged=size_averaged, reduced_form=reduced_form)] if GorD \
            else []
        if GorD:
            ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']
        else:
            ls_names = []
        # ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_L1']
    return ls_fn, ls_names


def pccloss(X, y, computer=ttools.get_ls_fn('pcc'), lambda_pcc=100):
    return computer(X, y) * lambda_pcc


def pcc_ls_fn(is_train, GorD, **kwargs):
    # 处理关键词参数
    # try:
    #     # kwargs = args[0]
    #     gan_mode = kwargs.pop('gan_mode', 'lsgan')
    #     reduced_form = kwargs.pop('reduced_form', False)
    #     size_averaged = kwargs.pop('size_averaged', True)
    #     lambda_PCC = kwargs.pop('lambda_pcc', 1.)
    # except IndexError:
    #     # 如果没有传递参数
    #     gan_mode = 'lsgan'
    #     reduced_form = False
    #     lambda_PCC = 1.
    #     kwargs = {}
    #     size_averaged = True
    gan_mode = kwargs.pop('gan_mode', 'lsgan')
    reduced_form = kwargs.pop('reduced_form', False)
    size_averaged = kwargs.pop('size_averaged', True)
    lambda_PCC = kwargs.pop('lambda_pcc', 100.)
    device = kwargs.pop('device', torch.device('cpu'))
    # GAN、PCC不接受size_averaged参数
    if not size_averaged and 'reduction' not in kwargs.keys():
        kwargs['reduction'] = 'none'
    criterionGAN = ttools.get_ls_fn('gan', gan_mode=gan_mode, device=device, **kwargs)
    criterionPCC = ttools.get_ls_fn('pcc', **kwargs)
    # self.criterionGAN = lambda pred, target_is_real: criterionGAN(pred, target_is_real)
    criterionPCC = functools.partial(l1loss, computer=criterionPCC, lambda_l1=lambda_PCC)

    if is_train:
        ls_fn = [functools.partial(cGAN_G_ls_fn, criterionGAN=criterionGAN, criterionL1=criterionPCC,
                                   size_averaged=size_averaged, reduced_form=reduced_form)] if GorD \
            else [functools.partial(D_ls_fn, criterionGAN=criterionGAN, reduced_form=reduced_form)]
        if GorD:
            ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_PCC']
        else:
            ls_names = ['D_LS'] if reduced_form else ['D_LS', 'D_real', 'D_fake']
    else:
        ls_fn = [functools.partial(cGAN_G_ls_fn, criterionGAN=criterionGAN, criterionL1=criterionPCC,
                                   size_averaged=size_averaged, reduced_form=reduced_form)] if GorD \
            else []
        if GorD:
            ls_names = ['G_LS'] if reduced_form else ['G_LS', 'G_GAN', 'G_PCC']
        else:
            ls_names = []
    return ls_fn, ls_names


def PCC_G_ls_fn(X, y, pred,
                netD, criterionGAN, criterionPCC,
                size_averaged=True, reduced_form=False):
    # 首先，G(A)需要骗过分辨器
    fake_AB = torch.cat((X, pred), 1)
    pred_fake = netD(fake_AB)
    gan_ls = criterionGAN(pred_fake, True)
    pcc_ls = criterionPCC(pred, y)
    if not size_averaged:
        gan_ls = gan_ls.mean(dim=list(range(len(pred_fake.shape)))[1:])
        # pcc_ls = pcc_ls.mean(dim=list(range(len(pred.shape)))[1:])
    if reduced_form:
        return gan_ls + pcc_ls,
    else:
        return gan_ls + pcc_ls, gan_ls, pcc_ls


def cGAN_G_ls_fn(X, y, pred,
                 netD, criterionGAN, criterionL1,
                 size_averaged=True, reduced_form=False):
    # 首先，G(A)需要骗过分辨器
    fake_AB = torch.cat((X, pred), 1)
    pred_fake = netD(fake_AB)
    gan_ls = criterionGAN(pred_fake, True)
    l1_ls = criterionL1(pred, y)
    if not size_averaged:
        gan_ls = gan_ls.mean(dim=list(range(len(pred_fake.shape)))[1:])
        l1_ls = l1_ls.mean(dim=list(range(len(pred.shape)))[1:])
    if reduced_form:
        return gan_ls + l1_ls,
    else:
        return gan_ls + l1_ls, gan_ls, l1_ls


def D_ls_fn(X, y, pred,
            netD, criterionGAN, reduced_form=False):
    fake_AB = torch.cat((X, pred), 1)
    pred_fake = netD(fake_AB.detach())
    fake_ls = criterionGAN(pred_fake, False)
    # 真实值
    real_AB = torch.cat((X, y), 1)
    pred_real = netD(real_AB)
    real_ls = criterionGAN(pred_real, True)
    # 组合损失值并计算梯度，0.5是两个预测值的权重
    ls = (fake_ls + real_ls) * 0.5
    if reduced_form:
        return ls,
    else:
        return ls, real_ls, fake_ls


def _get_lr_scheduler(optimizer, *args):
    """获取一个学习率规划器

    对于 "linear" 类的规划器，设定为在 “n_epochs” 世代内保持原有学习率，在此后的 “n_epochs_decay” 线性衰减至0。
    对于其他的规划器(step, plateau以及cosine)，使用默认的Pytorch规划器

    :param args: 每个网络的优化器对应的学习率规划器的参数。
        需要为可迭代对象，元素的个数等于本网络持有的优化器数目，每个元素按照位序为优化器分配学习率规划器。
        args的每个元素均为一个二元组，[0]为规划器类型，[1]为规划器关键字参数
    """
    # 参数数量判断
    if len(args) == 0:
        return []
    elif len(args) > 1:
        raise ValueError(f"Pix2Pix的子模块只支持一组学习率规划器参数，然而却收到了{len(args)}组！")
    else:
        ss, kwargs = check_prepare_args(*args[0])
    # 设置默认值
    if ss == 'linear':
        epoch_count = 1 if 'epoch_count' not in kwargs.keys() else kwargs['epoch_count']
        n_epochs_decay = 100 if 'n_epochs_decay' not in kwargs.keys() else kwargs['n_epochs_decay']
        n_epochs = 100 if 'n_epochs' not in kwargs.keys() else kwargs['n_epochs']
        kwargs.update({'epoch_count': epoch_count, 'n_epochs_decay': n_epochs_decay, "n_epochs": n_epochs})
    elif ss == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l

        lr_lambda = lambda_rule if 'lr_lambda' not in kwargs.keys() else kwargs['lr_lambda']
        kwargs.update({'lr_lambda': lr_lambda})
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
    return [ttools.get_lr_scheduler(optimizer, ss, **kwargs)]
    # scheduler_s = []
    # for (ss, kwargs), optimizer in zip_longest(args, optimizer, fillvalue=(None, None)):
    #     if ss is None or optimizer == (None, None):
    #         break
    #     if ss == 'linear':
    #         epoch_count = 1 if 'epoch_count' not in kwargs.keys() else kwargs['epoch_count']
    #         n_epochs_decay = 100 if 'n_epochs_decay' not in kwargs.keys() else kwargs['n_epochs_decay']
    #         n_epochs = 100 if 'n_epochs' not in kwargs.keys() else kwargs['n_epochs']
    #
    #         def lambda_rule(epoch):
    #             lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    #             return lr_l
    #
    #         ss = 'lambda'
    #         kwargs['lr_lambda'] = lambda_rule
    #     elif ss == 'step':
    #         step_size = 50 if 'step_size' not in kwargs.keys() else kwargs['step_size']
    #         gamma = 0.1 if 'gamma' not in kwargs.keys() else kwargs['gamma']
    #         kwargs.update({'step_size': step_size, 'gamma': gamma})
    #     elif ss == 'plateau':
    #         mode = 'min' if 'mode' not in kwargs.keys() else kwargs['mode']
    #         factor = 0.2 if 'factor' not in kwargs.keys() else kwargs['factor']
    #         threshold = 0.01 if 'threshold' not in kwargs.keys() else kwargs['threshold']
    #         patience = 5 if 'patience' not in kwargs.keys() else kwargs['patience']
    #         kwargs.update({'mode': mode, 'factor': factor, 'threshold': threshold, 'patience': patience})
    #     elif ss == 'cosine':
    #         T_max = 100 if 'T_max' not in kwargs.keys() else kwargs['T_max']
    #         eta_min = 0 if 'eta_min' not in kwargs.keys() else kwargs['eta_min']
    #         kwargs.update({'T_max': T_max, 'eta_min': eta_min})
    #     scheduler_s.append(ttools.get_lr_scheduler(optimizer, ss, **kwargs))
    # return scheduler_s


def _get_optimizer(module, *args):
    """获取pix2pix的优化器。
    目前只允许给netG，netD分配优化器，且每个网络只能分配一个。

    :param args: 各个优化器的类型以及关键字参数，如果指定数目少于2个，则会用论文中的参数填充。
    :return: 优化器序列，学习率名称序列
    """
    # # 如果指定参数少于2个，则使用默认值填充至2个
    # if len(args) < 2:
    #     # 如果优化器类型指定太少，则使用默认值补充
    #     args = (*args, *[('adam', {}) for _ in range(2 - len(args))])
    assert len(args) == 1, f"{module.__class__.__name__}只接受一组优化器参数，然而收到了{len(args)}组参数！"
    type_s, kwargs = check_prepare_args(*args[0])
    return [ttools.get_optimizer(module, type_s, **kwargs)], [f"{module.__class__.__name__[-1]}_LRs"]
    # # 检查优化器参数
    # optimizers = []
    # for (type_s, kwargs), net in zip(args, [self.netG, self.netD]):
    #     # 设置默认值
    #     if 'lr' not in kwargs.keys():
    #         kwargs['lr'] = 0.2
    #     if 'betas' not in kwargs.keys():
    #         kwargs['betas'] = (0.1, 0.999)
    #     optimizers.append(ttools.get_optimizer(net, type_s, **kwargs))
    # return optimizers, ['G_lrs', 'D_lrs']


def _backward_impl(main_ls):
    main_ls[0].backward()


from .pix2pix import Pix2Pix
