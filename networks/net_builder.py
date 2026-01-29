import re
import warnings
from collections import OrderedDict
from itertools import zip_longest
from typing import Iterable

import torch
from torchsummary import summary

from networks import BasicNN, net_predict_state, net_finetune_state, net_train_state, net_idle_state, get_net_state
from utils import ptools, ttools


supported_usage = ["predict", "train", "finetune"]


class NetBuilder:
    """
    NetBuilder负责保存BasicNN的位置参数和关键字参数，以及准备参数

    Args:

    """

    def __init__(self, module, **config):
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
            msg = (f"输入的参数不完整，只收到了{n_valid_argments}组有效参数！"
                   f"{module.__name__}的构造参数包括位置参数{m_po}, 位置/关键字参数{m_pok}, "
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
        """
        根据参数构造一个神经网络

        :param usage: 构造网络的用途。如果并非构建空闲网络，则会进行网络的激活
        :param mute: 如果为True，则会在命令行中输出构造进度和网络信息
        :return: 构造完成的神经网络
        """
        # if not usage:
        #     usage = self.usage
        if not mute:
            print(f'\r正在构造{self.module.__name__}', end='', flush=True)
        assert hasattr(self, 'init_args'), (f"没有为网络配置初始化位置参数，请通过调用{self.__class__.__name__}()."
                                            f"{self.set_module_init_args.__name__}()方法进行配置！")
        assert hasattr(self, 'init_kwargs'), (f"没有为网络配置初始化关键字参数，请通过调用{self.__class__.__name__}()."
                                              f"{self.set_module_init_args.__name__}()方法进行配置！")
        try:
            # 提取预处理参数
            init_meth = self.config.get("init_meth", "zero")
            init_meth_kwargs = self.config.get("init_kwargs", {})
            if init_meth == "entire_nn":
                # 如果发现需要加载一整个网络对象，则直接通过torch.load()加载，并跳过init_net
                where = init_meth_kwargs.pop('where')
                net = torch.load(where)
            else:
                # 否则调用初始化函数进行网络对象的创建，并进行初始化
                net = self.module(*self.init_args, **self.init_kwargs)
                self.init_net(net, init_meth, **init_meth_kwargs)
        except TypeError as e:
            pattern = r"takes from (\d+) to (\d+) positional arguments but (\d+) were given"
            if re.search(pattern, str(e)):
                raise TypeError(f"网络创建参数错误，请参考接口说明："
                                f"{self.module.__init__.__doc__}")
            else:
                raise e
        except FileNotFoundError:
            raise FileNotFoundError(f'找不到网络文件{where}！')
        if not mute:
            self.__list_net(net)
            print(f'\r构造{self.module.__name__}完成')
        if self.usage != net_idle_state:
            self.activate_model(net, self.usage, mute)
        return net

    def init_net(self, net, init_str, **kwargs):
        """初始化各模块参数。
        该方法会使用init_str所指初始化方法初始化网络net。
        init_str赋值为"state"时，启用预训练模型加载，使用where参数指定的.ptsd文件加载预训练参数，
        init_str赋值为"entire_nn"时，启用预训练模型加载，目前尚未实现整个网络的预加载。
        init_str赋值为"self_define"时，启用自定义的初始化方法，逐层遍历进行模型参数加载：
            须在关键词参数中通过“init_fn”参数指定自定义的初始化方法，且方法的签名需为：
            def fn(module, prefix, **kwargs) -> None
                :param module: 进行初始化的层
                :param prefix: 通过“.”进行分隔的层级信息
                :param kwargs: _init_submodules()方法接收到的kwargs参数，已经排除了init_fn参数
        其他init_str参数使用pytorch提供的官方方法进行初始化

        :param net: 需要进行初始化的网络
        :param init_str: 初始化方法类型
        :param kwargs: 初始化方法参数
        :return: None
        """
        if init_str == "state":
            try:
                where = kwargs.pop('where')
                if isinstance(OrderedDict, where):
                    net.load_state_dict(where)
                elif isinstance(str, where):
                    where = torch.load(where) if self.device.type == "cuda" else \
                        torch.load(where, map_location=torch.device('cpu'), weights_only=True)
                else:
                    raise ValueError(f"{init_str}初始化网络时，不支持处理{type(where)}！")
                net.load_state_dict(where)
            except KeyError:
                raise ValueError('选择预训练好的参数初始化网络，需要使用where关键词提供字典参数OrderedDict()或者模型的路径str！')
        elif init_str == "self_define":
            try:
                fn = kwargs.pop("init_fn")
                assert callable(fn), ("init_kwargs参数列表中的init_fn参数需要为可调用对象，用于指定初始化的具体实现。"
                                      "该可调用对象的签名为def fn(module, prefix, **kwargs) -> None)，"
                                      "module为nn.Sequential作为迭代器后每次取到的模块，prefix为该模块在内部的名称，"
                                      "kwargs为其他init_kwargs")
            except KeyError:
                raise KeyError('自定义初始化方法，需要给定网络创建关键字参数init_kwargs，其字段需要包括init_fn参数，'
                               '通过该参数中指定可调用对象。')

            def load(module, prefix=''):
                for name, child in module._modules.items():
                    if child is not None:
                        child_prefix = prefix + name + '.'
                        load(child, child_prefix)
                        fn(module, prefix, **kwargs)

            load(net)
        else:
            init_fn = ttools.init_wb(init_str, **kwargs)
            net.apply(init_fn)

    def activate_model(self, net, usage, mute: bool = False):
        """
        对输入的net对象进行激活，遍历net包含的BasicNN模块进行激活。
        激活操作是指对这些模块赋值优化器、学习率规划器、训练和测试损失函数。

        :param net: 可迭代对象，找出其中的BasicNN模块进行激活操作。
        :param mute: 是否进行激活进度提示
        """
        assert usage in [net_train_state, net_predict_state, net_finetune_state], "网络用于训练、预测和微调时才需要进行激活！"
        # 对准备参数进行检查
        o_args, l_args, tr_ls_args, ts_ls_args = (self._optimizer_args, self._lr_scheduler_args,
                                                  self._training_ls_fn_args, self._testing_ls_fn_args)
        assert isinstance(o_args, Iterable), "优化器参数需要为可迭代对象，每个元素对应一个基础网络的优化器参数！"
        assert isinstance(l_args, Iterable), "学习率规划器参数需要为可迭代对象，每个元素对应一个基础网络的学习率规划器参数！"
        assert isinstance(tr_ls_args, Iterable), "训练损失函数参数需要为可迭代对象，每个元素对应一个基础网络的训练损失函数参数！"
        assert isinstance(ts_ls_args, Iterable), "测试损失函数参数需要为可迭代对象，每个元素对应一个基础网络的测试损失函数参数！"
        # 提取出本网络中的所有BasicNN，并对它们进行准备参数设置，即通过activate()进行激活
        if not mute: print("依次对", end="")
        bnn_s = list(filter(lambda m: isinstance(m, BasicNN), reversed(list(net.modules()))))
        for bnn, o, l, tr, ts in zip_longest(bnn_s, o_args, l_args, tr_ls_args, ts_ls_args, fillvalue=[]):
            if not mute: print(bnn.__class__.__name__, end=" ")
            if not bnn: raise ValueError(f"赋值的参数比赋值的网络数要多！可赋值的网络总共包括："
                                         f"{', '.join(map(lambda m: m.__class__.__name__, bnn_s))}")
            bnn.activate(usage, o, l, tr, ts)
        if not mute: print(f"进行{usage}激活", flush=True, end="")

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
    def device(self):
        return torch.device(self.config['device'])

    @property
    def usage(self):
        return get_net_state(self.config['usage'])

    @usage.setter
    def usage(self, value):
        # 对用途进行状态转换
        self.config['usage'] = get_net_state(value)
        # if value == "predict":
        #     self.config['usage'] = net_predict_state
        # elif value == "train":
        #     self.config['usage'] = net_train_state
        # elif value == "finetune":
        #     self.config['usage'] = net_finetune_state
        # else:
        #     raise ValueError(f"不支持将用途设置为{value}！支持的状态包括：{supported_usage}")


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
