from itertools import zip_longest
from typing import Iterable


def check_prepare_args(*args):
    assert len(args) == 2, f"准备元组参数长度需要为2，分别为类型字符串和构造关键字参数。收到的参数数量为{len(args)}！"
    assert isinstance(args[0], str) and isinstance(args[1], dict), \
        f"准备元组参数分别为类型字符串和构造关键字参数，收到的参数类型为{type(args[0])}和{type(args[1])}！"
    return args


from .basic_nn import BasicNN


def configure_network(module: BasicNN, is_train: bool, o_args=None, l_args=None, tr_ls_args=None, ts_ls_args=None):
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
    # if (isinstance(o_args[0], tuple) and isinstance(l_args[0], tuple) and
    #     isinstance(tr_ls_args[0], tuple) and isinstance(ts_ls_args[0], tuple)):
    #     # 如果第一个参数是元组，那么说明用户只为本网络准备参数
    #     # 将第一组参数视为给本基础网络设置的参数
    #     self_o, self_l, self_trl, self_tsl = o_args, l_args, tr_ls_args, ts_ls_args
    #     o_args, l_args, tr_ls_args, ts_ls_args = None, None, None, None
    #     # self.optimizer_s, self.lr_names = self._get_optimizer(*o_args)
    #     # self.scheduler_s = self._get_lr_scheduler(*l_args)
    #     # self.train_ls_fn_s, self.train_ls_names = self._get_ls_fn(*tr_ls_args)
    #     # self.test_ls_fn_s, self.test_ls_names = self._get_ls_fn(*ts_ls_args)
    #     # try:
    #     #     # 设置梯度裁剪方法
    #     #     self._gradient_clipping = self._gradient_clipping
    #     # except AttributeError:
    #     #     self._gradient_clipping = None
    #     # self.ready = True
    #     # return
    # elif (isinstance(o_args[0], list) and isinstance(l_args[0], list) and
    #         isinstance(tr_ls_args[0], list) and isinstance(ts_ls_args[0], list)):
    #     # 如果第一个参数是列表，那么说明用户有为子网络准备参数
    #     o_args, l_args, tr_ls_args, ts_ls_args = iter(o_args), iter(l_args), iter(tr_ls_args), iter(ts_ls_args)
    #     self_o, self_l, self_trl, self_tsl = next(o_args), next(l_args), next(tr_ls_args), next(ts_ls_args)
    # else:
    #     raise ValueError(
    #             f"不支持的准备参数组合{type(o_args[0]).__name__}&{type(l_args[0]).__name__}&"
    #             f"{type(tr_ls_args[0]).__name__}&{type(ts_ls_args[0]).__name__}！"
    #             f"准备参数应为全列表，表示基本子网络的准备参数组；或者全元组，表示本基本网络的准备参数组。"
    #         )
    # 提取出本网络中的所有基础网络
    print("依次对", end="")
    for bnn, o, l, tr, ts in zip_longest(filter(lambda m: isinstance(m, BasicNN), reversed(list(module.modules()))),
                                         o_args, l_args, tr_ls_args, ts_ls_args, fillvalue=[]):
        print(bnn.__class__.__name__, end=" ")
        if bnn == []:
            raise ValueError(f"赋值的参数比赋值的网络数要多！可赋值的网络总共包括：{list(reversed(list(module.modules())))}")
        bnn.activate(is_train, o, l, tr, ts)
    print("进行初始化")


from .trainer import Trainer
