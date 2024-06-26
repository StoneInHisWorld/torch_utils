import torch
from torch import cuda, nn as nn
from torch.nn import init as init

from networks.layers.ganloss import GANLoss
from networks.layers.pcc import PCCLoss
from networks.layers.ssim import SSIMLoss

loss_es = ["l1", "entro", "mse", "huber", "ssim", "pcc", 'gan']
init_funcs = ["normal", "xavier", "zero", "state", 'constant', 'trunc_norm']
optimizers = ["sgd", "asgd", "adagrad", "adadelta", "rmsprop", "adam", "adamax"]
activations = ['sigmoid', 'relu', 'lrelu', 'tanh']
lr_schedulers = ["lambda", "step", 'constant', 'multistep', 'cosine', 'plateau']


def try_gpu(i=0):
    """
    获取一个GPU
    :param i: GPU编号
    :return: 第i号GPU。若GPU不可用，则返回CPU
    """
    if cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


# def get_optimizer(net: torch.nn.Module, optim_str, lr=0.1, w_decay=0., momentum=0.):
def get_optimizer(net: torch.nn.Module, optim_str='adam', lr=0.1, w_decay=0., **kwargs):
    if optim_str == "asgd":
        # 使用随机平均梯度下降优化器
        return torch.optim.ASGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    elif optim_str == "adagrad":
        # 使用自适应梯度优化器
        return torch.optim.Adagrad(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    elif optim_str == "adadelta":
        # 使用Adadelta优化器，Adadelta是Adagrad的改进
        return torch.optim.Adadelta(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    elif optim_str == "adam":
        # 使用Adaptive Moment Estimation优化器。Adam是RMSprop的改进。
        return torch.optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    elif optim_str == "adamax":
        # 使用Adamax优化器，Adamax是Adam的改进
        return torch.optim.Adamax(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    if optim_str == "rmsprop":
        # 使用RMSprop优化器，RMSprop是Adagrad的改进
        return torch.optim.RMSprop(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    elif optim_str == "sgd":
        # 使用随机梯度下降优化器
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            **kwargs
        )
    else:
        raise NotImplementedError(f"不支持优化器{optim_str}, 支持的优化器包括{optimizers}")


def get_ls_fn(ls_str: str = "mse", **kwargs):
    """获取损失函数。
    此处返回的损失函数不接收size_average参数，若需要非批量平均化的损失值，请指定reduction='none'
    :param ls_str: 损失函数对应字符串
    :param kwargs: 输入到损失值计算模块中的关键词参数。请注意，每个损失值计算模块的关键词参数可能不同！建议输入关键词参数时只选用一种损失值计算模块。
    :return: 损失函数模块
    """
    if ls_str == "l1":
        return nn.L1Loss(**kwargs)
    elif ls_str == "entro":
        return nn.CrossEntropyLoss(**kwargs)
    elif ls_str == "mse":
        return nn.MSELoss(**kwargs)
    elif ls_str == "huber":
        return nn.HuberLoss(**kwargs)
    elif ls_str == "ssim":
        return SSIMLoss(**kwargs)
    elif ls_str == 'pcc':
        return PCCLoss(**kwargs)
    elif ls_str == 'gan':
        return GANLoss(**kwargs)
    else:
        raise NotImplementedError(f"不支持损失函数{ls_str}, 支持的损失函数包括{loss_es}")


def init_wb(func_str: str = "xavier", **kwargs):
    """
    返回初始化权重、偏移参数的函数。
    :param func_str: 指定初始化方法的字符串
    :return: 包装好可直接调用的初始化函数
    """
    if func_str == 'state':
        # 加载预先加载的网络
        return None
    elif func_str == "normal":
        mean, std = kwargs.pop('mean', 0), kwargs.pop('std', 1)
        w_init = lambda m: init.normal_(m, mean, std)
        b_init = lambda m: init.normal_(m, mean, std)
    elif func_str == "xavier":
        w_init, b_init = init.xavier_uniform_, init.zeros_
    elif func_str == "zero":
        w_init, b_init = init.zeros_, init.zeros_
    elif func_str == 'constant':
        w_value, b_value = kwargs.pop('w_value', 1), kwargs.pop('b_value', 0)
        w_init = lambda m: init.constant_(m, w_value)
        b_init = lambda m: init.constant_(m, b_value)
    elif func_str == 'trunc_norm':
        mean, std = kwargs.pop('mean', 0), kwargs.pop('std', 1)
        a, b = kwargs.pop('a', 0), kwargs.pop('b', 1)
        w_init = lambda m: init.trunc_normal_(m, mean, std, a, b)
        b_init = init.zeros_
    else:
        raise NotImplementedError(f"不支持的初始化方式{func_str}, 当前支持的初始化方式包括{init_funcs}")

    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if m.weight is not None:
                w_init(m.weight)
            if m.bias is not None:
                b_init(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if func_str != 'xavier':
                w_init(m.weight)
                b_init(m.bias)
        else:
            return

    return _init


def get_lr_scheduler(optimizer, which: str = 'step', **kwargs):
    """获取学习率规划器
    :param optimizer: 指定规划器的学习率优化器对象。
    :param which: 使用哪种类型的规划器
    :param kwargs: 指定规划器的关键词参数
    :return: 规划器对象。
    """
    if which == 'step':
        if kwargs == {}:
            kwargs = {'step_size': 100, 'gamma': 1}
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif which == 'lambda':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, **kwargs)
    elif which == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif which == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif which == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif which == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif which == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"不支持的学习率规划器{which}, 当前支持的初始化方式包括{lr_schedulers}")


def get_activation(which: str = 'step', **kwargs):
    """获取激活函数
    :param which: 使用哪种类型的激活函数
    :param kwargs: 指定激活函数的关键词参数
    :return: 激活函数层。
    """
    if which == 'sigmoid':
        return torch.nn.Sigmoid()
    elif which == 'relu':
        return torch.nn.ReLU(**kwargs)
    elif which == 'lrelu':
        return torch.nn.LeakyReLU(**kwargs)
    elif which == 'tanh':
        return torch.nn.Tanh()
    else:
        raise NotImplementedError(f"不支持的激活函数{which}, 当前支持的激活函数层包括{activations}")


def sample_wise_ls_fn(x, y, ls_fn):
    """计算每个样本的损失值的损失函数
    :param x: 特征集
    :param y: 标签集
    :param ls_fn: 基础损失函数
    :return: 包装好的平均损失函数
    """
    ls = ls_fn(x, y)
    return ls.mean(dim=list(range(len(ls.shape)))[1:])