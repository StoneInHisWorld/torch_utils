import warnings
from itertools import zip_longest
from typing import Iterable

from torchsummary import summary

from networks import BasicNN, net_predict_state, net_finetune_state, net_train_state
from utils import ptools


supported_usage = ["predict", "train", "finetune"]


class NetBuilder:
    """
    NetBuilder负责保存BasicNN的位置参数和关键字参数，以及准备参数

    Args:


    Usage:
        nb = NetBuilder(*base_args, **base_kwargs)
        nb.set_optimizer_args('sub1', lr=1e-3, weight_decay=1e-5)
        nb.set_lr_scheduler_args('sub1', step_size=10, gamma=0.1)
        args, kwargs = nb.prepare_args('sub1')
        model = nb.build(BasicNNClass, 'sub1') # equivalent to BasicNNClass(*args, **kwargs)
    """

    def __init__(self, module, config):
        self.module = module
        self.config = config
        self._optimizer_args = []
        self._lr_scheduler_args = []
        self._training_ls_fn_args = []
        self._testing_ls_fn_args = []

    def set_module_init_args(self, *init_args, **init_kwargs):
        module = self.module
        m_po, m_pok, m_ko, m_needed = ptools.get_signature(module)
        _, bn_pok, bn_ko, _ = ptools.get_signature(BasicNN)
        n_valid_argments = len(init_args) + len(init_kwargs.keys() - [b[0] for b in bn_ko])
        if n_valid_argments < len(m_needed):
            msg = (f"输入的参数不完整！{module.__name__}的构造参数包括位置参数{m_po}, 位置/关键字参数{m_pok}, "
                   f"关键字参数{m_ko}。\n{BasicNN.__name__}可以输入的参数包括{bn_ko}。\n"
                   f"创建{module.__name__}必需的参数为{m_needed}。")
            raise ValueError(msg)

        assert issubclass(module, BasicNN), f'{self.__class__.__name__}只服务BasicNN的子类！'
        self.module = module
        self.init_args = init_args
        init_kwargs['with_checkpoint'] = self.config['with_checkpoint']
        init_kwargs["device"] = self.config["device"]
        self.init_kwargs = init_kwargs

    # Setter methods for per-submodule argument groups
    def set_optimizer_args(self, *args):
        """保存模型的优化器参数"""
        self._optimizer_args = args

    def set_lr_scheduler_args(self, *args):
        """保存模型的学习率规划期参数"""
        self._lr_scheduler_args = args

    def set_training_ls_fn_args(self, *args):
        """保存模型的训练损失函数参数"""
        self._training_ls_fn_args = args

    def set_testing_ls_fn_args(self, *args):
        """保存模型的测试损失函数参数"""
        self._testing_ls_fn_args = args

    def build(self, mute: bool = False):
        """根据参数构造一个神经网络
        :param net_class: 构造的神经网络类型，作为类构造器传入
        :param n_init_args: 类构造器所用位置参数
        :param n_init_kwargs: 类构造器所用关键字参数
        :return: 构造完成的神经网络
        """
        if not mute:
            print(f'\r正在构造{self.module.__name__}', end='', flush=True)
        assert hasattr(self, 'init_args'), (f"没有为网络配置初始化位置参数，请通过调用{self.__class__.__name__}()."
                                            f"{self.set_module_init_args.__name__}()方法进行配置！")
        assert hasattr(self, 'init_kwargs'), (f"没有为网络配置初始化关键字参数，请通过调用{self.__class__.__name__}()."
                                              f"{self.set_module_init_args.__name__}()方法进行配置！")
        net = self.module(*self.init_args, **self.init_kwargs)
        if not mute:
            print(f'\r构造{self.module.__name__}完成')
            self.__list_net(net)
        self.activate_model(net, mute)
        return net

    def activate_model(self, net, mute: bool = False):
        """<UNK>
        :param net: <UNK>
        """
        # # 对用途进行状态转换
        # usage = self.config["usage"]
        # if usage == "predict":
        #     usage = net_predict_state
        # elif usage == "train":
        #     usage = net_train_state
        # elif usage == "finetune":
        #     usage = net_finetune_state
        # else:
        #     raise ValueError(f"对网络进行激活时不支持将网络状态设置为{usage}！支持的状态包括：{supported_usage}")
        # 对准备参数进行检查
        o_args, l_args, tr_ls_args, ts_ls_args = (self._optimizer_args, self._lr_scheduler_args,
                                                  self._training_ls_fn_args, self._testing_ls_fn_args)
        assert isinstance(o_args, Iterable), "优化器参数需要为可迭代对象，每个元素对应一个基础网络的优化器参数！"
        assert isinstance(l_args, Iterable), "学习率规划器参数需要为可迭代对象，每个元素对应一个基础网络的学习率规划器参数！"
        assert isinstance(tr_ls_args, Iterable), "训练损失函数参数需要为可迭代对象，每个元素对应一个基础网络的训练损失函数参数！"
        assert isinstance(ts_ls_args, Iterable), "测试损失函数参数需要为可迭代对象，每个元素对应一个基础网络的测试损失函数参数！"
        # 提取出本网络中的所有BasicNN，并对它们进行准备参数设置
        if not mute:
            print("依次对", end="")
        bnn_s = list(filter(lambda m: isinstance(m, BasicNN), reversed(list(net.modules()))))
        for bnn, o, l, tr, ts in zip_longest(bnn_s, o_args, l_args, tr_ls_args, ts_ls_args, fillvalue=[]):
            if not mute:
                print(bnn.__class__.__name__, end=" ")
            if not bnn:
                raise ValueError(f"赋值的参数比赋值的网络数要多！"
                                 f"可赋值的网络总共包括：{', '.join(map(lambda m: m.__class__.__name__, bnn_s))}")
            bnn.activate(self.usage, o, l, tr, ts)
        if not mute:
            print(f"进行{self.usage}初始化", flush=True, end="")
        # configure_network(
        #     net, usage, mute, self._optimizer_args, self._lr_scheduler_args,
        #     self._training_ls_fn_args, self._testing_ls_fn_args
        # )

    def __list_net(self, net) -> None:
        """打印网络信息。
        :param net: 待打印的网络
        :return: None
        """
        if self.config['print_net']:
            input_size = net.input_size
            if input_size:
                try:
                    summary(net, input_size=(self.config['batch_size'], *input_size), device=net.device)
                    return
                except Exception as e:
                    warnings.warn(f"打印网络时遇到错误：{e}，只显示网络结构！")
            else:
                warnings.warn(f"输入形状{input_size}无法解析，只显示网络结构！")
            print(net)

    @property
    def usage(self):
        return self.config['usage']

    @usage.setter
    def usage(self, value):
        # 对用途进行状态转换
        if value == "predict":
            self.config['usage'] = net_predict_state
        elif value == "train":
            self.config['usage'] = net_train_state
        elif value == "finetune":
            self.config['usage'] = net_finetune_state
        else:
            raise ValueError(f"不支持将用途设置为{value}！支持的状态包括：{supported_usage}")


# def configure_network(
#     module: BasicNN, usage: str, mute: bool = False,
#     o_args=None, l_args=None, tr_ls_args=None, ts_ls_args=None
# ):
#     """训练准备实现
#     获取优化器（对应学习率名称）、学习率规划器以及损失函数（训练、测试损失函数名称），储存在自身对象中。
#     获取顺序是先子网络，后主网络
#     :param o_args: 优化器参数列表。参数列表中每一项对应一个优化器设置，每一项签名均为(str, dict)，
#         str指示优化器类型，dict指示优化器构造关键字参数。
#     :param l_args: 学习率规划器参数列表。参数列表中每一项对应一个学习率规划器设置，每一项签名均为(str, dict)，
#         str指示学习率规划器类型，dict指示学习率规划器构造关键字参数。
#     :param tr_ls_args: 训练损失函数参数列表。参数列表中每一项对应一个损失函数设置，每一项签名均为(str, dict)，
#         str指示损失函数类型，dict指示损失函数构造关键字参数。
#     :param ts_ls_args: 测试损失函数参数列表。参数列表中每一项对应一个损失函数设置，每一项签名均为(str, dict)，
#         str指示损失函数类型，dict指示损失函数构造关键字参数。
#     :return: None
#     """
#     # 此处的类型检查针对给定参数能否分配给不同的BasicNN
#     if ts_ls_args is None:
#         ts_ls_args = []
#     if tr_ls_args is None:
#         tr_ls_args = []
#     if l_args is None:
#         l_args = []
#     if o_args is None:
#         o_args = []
#     assert isinstance(o_args, Iterable), "优化器参数需要为可迭代对象，每个元素对应一个基础网络的优化器参数！"
#     assert isinstance(l_args, Iterable), "学习率规划器参数需要为可迭代对象，每个元素对应一个基础网络的学习率规划器参数！"
#     assert isinstance(tr_ls_args, Iterable), "训练损失函数参数需要为可迭代对象，每个元素对应一个基础网络的训练损失函数参数！"
#     assert isinstance(ts_ls_args, Iterable), "测试损失函数参数需要为可迭代对象，每个元素对应一个基础网络的测试损失函数参数！"
#     # 提取出本网络中的所有BasicNN
#     if not mute:
#         print("依次对", end="")
#     bnn_s = list(filter(lambda m: isinstance(m, BasicNN), reversed(list(module.modules()))))
#     for bnn, o, l, tr, ts in zip_longest(bnn_s, o_args, l_args, tr_ls_args, ts_ls_args, fillvalue=[]):
#         if not mute:
#             print(bnn.__class__.__name__, end=" ")
#         if not bnn:
#             raise ValueError(f"赋值的参数比赋值的网络数要多！"
#                              f"可赋值的网络总共包括：{', '.join(map(lambda m: m.__class__.__name__, bnn_s))}")
#         bnn.activate(usage, o, l, tr, ts)
#     if not mute:
#         print(f"进行{usage}初始化", flush=True, end="")
