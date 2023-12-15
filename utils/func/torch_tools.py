import torch
from torch import cuda, nn as nn
from torch.nn import init as init

from networks.layers.ssim import SSIMLoss

loss_es = ['l1', 'entro', 'mse', 'huber', 'ssim']
init_funcs = ['normal', 'xavier', 'zero']
# init_funcs = ['normal', 'xavier', 'zero', 'entire_trained', 'state_trained']
optimizers = ['sgd', 'asgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax']


def try_gpu(i=0):
    """
    获取一个GPU
    :param i: GPU编号
    :return: 第i号GPU。若GPU不可用，则返回CPU
    """
    if cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def get_optimizer(net: torch.nn.Module, optim_str, lr=0.1, w_decay=0., momentum=0.):
    assert optim_str in optimizers, f'不支持优化器{optim_str}, 支持的优化器包括{optimizers}'
    if optim_str == 'sgd':
        # 使用随机梯度下降优化器
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            momentum=momentum
        )
    elif optim_str == 'asgd':
        # 使用随机平均梯度下降优化器
        return torch.optim.ASGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'adagrad':
        # 使用自适应梯度优化器
        return torch.optim.Adagrad(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'adadelta':
        # 使用Adadelta优化器，Adadelta是Adagrad的改进
        return torch.optim.Adadelta(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'rmsprop':
        # 使用RMSprop优化器，RMSprop是Adagrad的改进
        return torch.optim.RMSprop(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            momentum=momentum
        )
    elif optim_str == 'adam':
        # 使用Adaptive Moment Estimation优化器。Adam是RMSprop的改进。
        return torch.optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )
    elif optim_str == 'adamax':
        # 使用Adamax优化器，Adamax是Adam的改进
        return torch.optim.Adamax(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
        )


def get_loss(loss_str: str = 'mse'):
    """
    获取损失函数。
    :param loss_str: 损失函数对应字符串
    :return: 损失函数模块
    """
    assert loss_str in loss_es, \
        f'不支持损失函数{loss_str}, 支持的损失函数包括{loss_es}'
    if loss_str == 'l1':
        return nn.L1Loss()
    elif loss_str == 'entro':
        return nn.CrossEntropyLoss()
    elif loss_str == 'mse':
        return nn.MSELoss()
    elif loss_str == 'huber':
        return nn.HuberLoss()
    elif loss_str == 'ssim':
        return SSIMLoss()


def init_wb(func_str: str = 'xavier'):
    """
    返回初始化权重、偏移参数的函数。
    :param func_str: 指定初始化方法的字符串
    :return: 包装好可直接调用的初始化函数
    """
    assert func_str in init_funcs, f'不支持的初始化方式{func_str}, 当前支持的初始化方式包括{init_funcs}'
    if func_str == 'normal':
        w_init = lambda m: init.normal_(m, 0, 1)
        b_init = lambda m: init.normal_(m, 0, 1)
    elif func_str == 'xavier':
        w_init, b_init = init.xavier_uniform_, init.zeros_
    # elif func_str == 'entire_trained':
    #     pass
    # elif func_str == 'state_trained':
    #     def _init(m: nn.Module) -> None:
    #         sd = torch.load(where)
    else:
        w_init, b_init = init.zeros_, init.zeros_

    def _init(m: nn.Module) -> None:
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            w_init(m.weight)
            b_init(m.bias)

    return _init
