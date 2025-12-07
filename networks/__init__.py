from itertools import zip_longest
from typing import Iterable


def check_prepare_args(who: type, *args):
    assert len(args) == 2, (f"准备元组参数长度需要为2，分别为类型字符串和构造关键字参数。"
                            f"{who.__name__}收到的参数数量为{len(args)}！")
    assert isinstance(args[0], str) and isinstance(args[1], dict), \
        (f"准备元组参数分别为类型字符串和构造关键字参数，"
         f"{who.__name__}收到的参数类型为{type(args[0])}和{type(args[1])}！")
    return args


from .basic_nn import BasicNN


# def configure_network(
#         module: BasicNN, is_train: bool,
#         o_args=None, l_args=None, tr_ls_args=None, ts_ls_args=None
# ):
def configure_network(
    module: BasicNN, is_train: bool, mute: bool = False,
    o_args=None, l_args=None, tr_ls_args=None, ts_ls_args=None
):
    """训练准备实现
    获取优化器（对应学习率名称）、学习率规划器以及损失函数（训练、测试损失函数名称），储存在自身对象中。
    获取顺序是先子网络，后主网络
    :param o_args: 优化器参数列表。参数列表中每一项对应一个优化器设置，每一项签名均为(str, dict)，
        str指示优化器类型，dict指示优化器构造关键字参数。
    :param l_args: 学习率规划器参数列表。参数列表中每一项对应一个学习率规划器设置，每一项签名均为(str, dict)，
        str指示学习率规划器类型，dict指示学习率规划器构造关键字参数。
    :param tr_ls_args: 训练损失函数参数列表。参数列表中每一项对应一个损失函数设置，每一项签名均为(str, dict)，
        str指示损失函数类型，dict指示损失函数构造关键字参数。
    :param ts_ls_args: 测试损失函数参数列表。参数列表中每一项对应一个损失函数设置，每一项签名均为(str, dict)，
        str指示损失函数类型，dict指示损失函数构造关键字参数。
    :return: None
    """
    if ts_ls_args is None:
        ts_ls_args = []
    if tr_ls_args is None:
        tr_ls_args = []
    if l_args is None:
        l_args = []
    if o_args is None:
        o_args = []
    assert isinstance(o_args, Iterable), "优化器参数需要为可迭代对象，每个元素对应一个基础网络的优化器参数！"
    assert isinstance(l_args, Iterable), "学习率规划器参数需要为可迭代对象，每个元素对应一个基础网络的学习率规划器参数！"
    assert isinstance(tr_ls_args, Iterable), "训练损失函数参数需要为可迭代对象，每个元素对应一个基础网络的训练损失函数参数！"
    assert isinstance(ts_ls_args, Iterable), "测试损失函数参数需要为可迭代对象，每个元素对应一个基础网络的测试损失函数参数！"
    # 提取出本网络中的所有基础网络
    if not mute:
        print("依次对", end="")
    bnn_s = list(filter(lambda m: isinstance(m, BasicNN), reversed(list(module.modules()))))
    for bnn, o, l, tr, ts in zip_longest(bnn_s, o_args, l_args, tr_ls_args, ts_ls_args, fillvalue=[]):
        if not mute:
            print(bnn.__class__.__name__, end=" ")
        if not bnn:
            raise ValueError(f"赋值的参数比赋值的网络数要多！"
                             f"可赋值的网络总共包括：{', '.join(map(lambda m: m.__class__.__name__, bnn_s))}")
        bnn.activate(is_train, o, l, tr, ts)
    if not mute:
        print("进行训练初始化" if is_train else "进行测试初始化", flush=True, end="")


from .trainer import Trainer
from .trainer import New2Trainer
from .net_builder import NetBuilder
from .nets import *