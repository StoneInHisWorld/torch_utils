import functools
import warnings
from functools import reduce
from typing import List, Tuple, Iterable

import torch
import torch.nn as nn
from torch.utils import checkpoint

from networks import check_prepare_args
from utils import ttools


class BasicNN(nn.Sequential):
    """基本神经网络

    提供神经网络的基本功能，包括训练准备（优化器生成、学习率规划器生成、损失函数生成）、模块初始化、前反向传播实现以及展示图片输出注释的实现。
    """

    def __init__(self, *args, **kwargs) -> None:
        """基本神经网络
        提供神经网络的基本功能，包括训练准备（优化器生成、学习率规划器生成、损失函数生成）、模块初始化、前反向传播实现以及展示图片输出注释的实现。

        :param args: 需要添加的网络层
        :param kwargs: 可选提供的参数。
        -- device: 网络所处设备。
        -- init_meth: 网络初始化方法。
        -- init_kwargs: 网络初始化方法所用参数。
        -- with_checkpoint: 是否使用检查点机制。
        -- input_size: 本网络指定的输入形状，赋值为除批量维的维度。例如赋值为（通道数，长，宽）或者（序列长度，）等。
        """
        # 设置默认值
        init_meth = 'zero' if 'init_meth' not in kwargs.keys() else kwargs['init_meth']
        device = torch.device('cpu') if 'device' not in kwargs.keys() else kwargs['device']
        with_checkpoint = False if 'with_checkpoint' not in kwargs.keys() else kwargs['with_checkpoint']
        init_kwargs = {} if 'init_kwargs' not in kwargs.keys() else kwargs['init_kwargs']
        self.input_size = None if 'input_size' not in kwargs.keys() else (-1, *kwargs.pop('input_size'))
        # 设置状态标志
        self._ready = False
        # 初始化各模块
        super(BasicNN, self).__init__(*args)
        self._init_submodules(init_meth, **init_kwargs)
        self._gradient_clipping = None
        # 设备迁移
        self.apply(lambda m: m.to(device))

        self._device = device
        if with_checkpoint:
            warnings.warn('使用“检查点机制”虽然会减少前向传播的内存使用，但是会大大增加反向传播的计算量！')
        self.__checkpoint = with_checkpoint

    def _get_optimizer(self, o_args) -> torch.optim.Optimizer or List[torch.optim.Optimizer]:
        """获取网络优化器
        根据参数列表中的每一项调用torch_tools.py中的get_optimizer()来获取优化器。
        每个优化器对应的学习率名称为LR_i，其中i对应优化器的序号。

        :param o_args: 每个优化器的构造参数，签名为(str, dict)，str指示优化器类型，dict指示优化器构造关键字参数。
        :return: (优化器序列，学习率名称)
        """
        optimizer_s, lr_names = [], []
        for i, i_args in enumerate(o_args):
            type_s, kwargs = check_prepare_args(self.__class__, *i_args)
            optimizer_s.append(ttools.get_optimizer(self, type_s, **kwargs))
            lr_names.append(f'LR_{i}')
        return optimizer_s, lr_names

    def _get_lr_scheduler(self, l_args):
        """为优化器定制学习率规划器
        根据参数列表中的每一项调用torch_tools.py中的get_lr_scheduler()来获取学习率规划器。

        :param l_args: 每个学习率规划器的构造参数，签名为(str, dict)，
            其中str指示学习率规划器类型，dict指示学习率规划器构造关键字参数。
        :return: 学习率规划器序列。
        """
        schedulers = []
        for optimizer, i_args in zip(self.optimizer_s, l_args):
            type_s, kwargs = check_prepare_args(self.__class__, *i_args)
            schedulers.append(ttools.get_lr_scheduler(optimizer, type_s, **kwargs))
        return schedulers

    def _get_ls_fn(self, ls_args):
        """获取网络的训练和测试损失函数序列，并设置网络的训练损失、测试损失名称
        根据参数列表中的每一项调用torch_tools.py中的get_ls_fn()来获取损失函数。
        获取损失函数前，会提取关键词参数中的size_averaged。size_averaged == True，则会返回torch_tools.sample_wise_ls_fn()包装过的损失函数。

        :param train_ls_args: 多个二元组，元组格式为（训练损失函数类型字符串，本损失函数关键词参数），若不指定关键词参数，请用空字典。
        :param test_ls_args: 多个二元组，元组格式为（测试损失函数类型字符串，本损失函数关键词参数），若不指定关键词参数，请用空字典。
        :return: 训练损失函数序列，训练损失函数名称，测试损失函数序列，测试损失函数名称
        """
        fn_s, name_s = [], []
        for i_args in ls_args:
            type_s, kwargs = check_prepare_args(self.__class__, *i_args)
            # 根据参数获取损失函数
            name_s.append(type_s.upper())
            size_averaged = kwargs.pop('size_averaged', True)
            unwrapped_fn = ttools.get_ls_fn(type_s, **kwargs)
            fn_s.append(
                unwrapped_fn if size_averaged else
                functools.partial(ttools.sample_wise_ls_fn, ls_fn=unwrapped_fn)
            )
        return fn_s, name_s

    def activate(self, is_train: bool,
                 o_args: Iterable, l_args: Iterable, tr_ls_args: Iterable,
                 ts_ls_args: Iterable):
        if self.ready:
            return
        if is_train:
            self.train()
            assert isinstance(o_args, Iterable), ("优化器参数需要为可迭代对象，其中的每个元素均为二元组，"
                                                  "二元组的0号位为优化器类型字符串，1号位为优化器构造关键字参数")
            self.optimizer_s, self.lr_names = self._get_optimizer(o_args)
            assert isinstance(l_args, Iterable), ("学习率规划器参数需要为可迭代对象，其中的每个元素均为二元组，"
                                                  "二元组的0号位为规划器类型字符串，1号位为规划器构造关键字参数")
            self.scheduler_s = self._get_lr_scheduler(l_args)
            assert isinstance(tr_ls_args, Iterable), ("训练损失函数参数需要为可迭代对象，其中的每个元素均为二元组，"
                                                      "二元组的0号位为损失函数类型字符串，1号位为损失函数构造关键字参数")
            self.train_ls_fn_s, self.train_ls_names = self._get_ls_fn(tr_ls_args)
        else:
            self.eval()
        assert isinstance(ts_ls_args, Iterable), ("测试损失函数参数需要为可迭代对象，其中的每个元素均为二元组，"
                                                  "二元组的0号位为损失函数类型字符串，1号位为损失函数构造关键字参数")
        self.test_ls_fn_s, self.test_ls_names = self._get_ls_fn(ts_ls_args)
        self._ready = True

    def _init_submodules(self, init_str, **kwargs):
        """初始化各模块参数。
        该方法会使用init_str所指初始化方法初始化所用层，若要定制化初始模块，请重载本函数。
        init_str赋值为"state"时，启用预训练模型加载，使用where参数指定的.ptsd文件加载预训练参数，
        init_str赋值为"entire_nn"时，启用预训练模型加载，目前尚未实现整个网络的预加载。
        init_str赋值为"self_define"时，启用自定义的初始化方法，逐层遍历进行模型参数加载：
            须在关键词参数中通过“init_fn”参数指定自定义的初始化方法，且方法的签名需为：
                def fn(module, prefix, **kwargs) -> None
                    :param module: 进行初始化的层
                    :param prefix: 通过“.”进行分隔的层级信息
                    :param kwargs: _init_submodules()方法接收到的kwargs参数，已经排除了init_fn参数
        其他init_str参数使用pytorch提供的官方方法进行初始化

        :param init_str: 初始化方法类型
        :param kwargs: 初始化方法参数
        :return: None
        """
        if init_str == "state":
            try:
                where = kwargs['where']
                paras = torch.load(where) if torch.cuda.is_available() else \
                    torch.load(where, map_location=torch.device('cpu'), weights_only=True)
                self.load_state_dict(paras)
            except IndexError:
                raise ValueError('选择预训练好的参数初始化网络，需要使用where关键词提供参数或者模型的路径！')
            except FileNotFoundError:
                raise FileNotFoundError(f'找不到网络参数文件{where}！')
        elif init_str == "entire_nn":
            raise NotImplementedError('针对预训练好的网络，请使用如下方法获取`net = torch.load("../xx.ptm")`')
        elif init_str == "self_define":
            try:
                fn = kwargs.pop("init_fn")
            except IndexError:
                raise ValueError('自定义初始化方法，需要在init_fn参数中指定可调用对象！')

            def load(module, prefix=''):
                for name, child in module._modules.items():
                    if child is not None:
                        child_prefix = prefix + name + '.'
                        load(child, child_prefix)
                        fn(module, prefix, **kwargs)

            load(self)
            del load
        else:
            init_fn = ttools.init_wb(init_str, **kwargs)
            self.apply(init_fn)

    def forward_backward(self, X, y, backward=True):
        """前向和反向传播。
        在进行前向传播后，会利用self.ls_fn()进行损失值计算，随后根据backward的值选择是否进行反向传播。
        若要更改optimizer.zero_grad()，optimizer.step()的操作顺序，请直接重载本函数。
        :param X: 特征集
        :param y: 标签集
        :param backward: 是否进行反向传播
        :return: 预测值，损失值集合
        """
        if backward:
            with torch.enable_grad():
                for optim in self.optimizer_s:
                    optim.zero_grad()
                result = self._forward_impl(X, y)
                self._backward_impl(*result[1])
                if self._gradient_clipping is not None:
                    self._gradient_clipping()
                for optim in self.optimizer_s:
                    optim.step()
            assert len(result) == 2, f'前反向传播需要返回元组（预测值，损失值集合），但实现返回的值为{result}'
            assert len(result[1]) == len(self.train_ls_names), \
                f'前向传播返回的损失值数量{len(result[1])}与指定的损失名称数量{len(self.train_ls_names)}不匹配。'
        else:
            with torch.no_grad():
                result = self._forward_impl(X, y)
            assert len(result) == 2, f'前向传播需要返回元组（预测值，损失值集合），但实现返回的值为{result}'
            assert len(result[1]) == len(self.test_ls_names), \
                f'前向传播返回的损失值数量{len(result[1])}与指定的损失名称数量{len(self.test_ls_names)}不匹配。'
        return result

    def _forward_impl(self, X, y) -> Tuple[torch.Tensor, List]:
        """前向传播实现。
        进行前向传播后，根据self._ls_fn()计算损失值，并返回。
        若要更改optimizer.zero_grad()以及backward()的顺序，请直接重载forward_backward()！
        :param X: 特征集
        :param y: 标签集
        :return: （预测值， （损失值集合））
        """
        pred = self(X)
        if torch.is_grad_enabled():
            ls_fn_s = self.train_ls_fn_s
        else:
            ls_fn_s = self.test_ls_fn_s
        return pred, [ls_fn(pred, y) for ls_fn in ls_fn_s]

    def _backward_impl(self, *ls_es):
        """反向传播实现
        可重载本函数实现定制的反向传播，默认对所有损失求和并反向传播
        :param ls_es: 损失值
        :return: None
        """
        assert len(ls_es) > 0, "反向传播没有收到损失值！"
        zeros = torch.zeros(1, requires_grad=True, device=ls_es[0].device)
        total = reduce(lambda x, y: x + y, ls_es, zeros)
        total.backward()

    def get_lr_groups(self):
        return self.lr_names, [optimizer.defaults['lr'] for optimizer in self.optimizer_s]

    def update_lr(self):
        for scheduler in self.scheduler_s:
            scheduler.step()

    def deactivate(self):
        # 清除训练痕迹
        if self.ready:
            for name in ["optimizer_s", "scheduler_s", "lr_names",
                         "train_ls_fn_s", "test_ls_fn_s", "train_ls_names",
                         "test_ls_names"]:
                if hasattr(self, name):
                    delattr(self, name)
        self._ready = False
        for bnn in filter(lambda m: isinstance(m, BasicNN), self.children()):
            bnn.deactivate()

    @property
    def ready(self):
        __ready = self._ready
        if __ready:
            for bnn in filter(lambda m: isinstance(m, BasicNN), self.children()):
                __ready = bnn.ready and __ready
        return __ready

    @property
    def device(self):
        return self._device

    def __str__(self):
        return ('网络结构：\n' + super().__str__() +
                '\n所处设备：' + str(self._device))

    def __call__(self, x):
        """调用函数
        集成了checkpoint机制的调用函数，如果启用checkpoint机制，则前向传播前会逐层检查checkpoint适用性。
        Checkpoint机制下，前向传播并不会保存反向传播梯度计算中所需的张量，反向传播中会重新计算，以此节省内存消耗，但是大大会增加计算量。

        :param x: 前向传播输入
        :return: 前向传播结果
        """
        # 如果指定了输入形状，则进行形状检查
        if self.input_size:
            # 排除掉批量大小维度，只检查通道维和长宽
            assert x.shape[1:] == self.input_size[1:], \
                f'输入网络的张量形状{x.shape}与网络要求形状{self.input_size}不匹配！'
        # checkpoint检查
        if self.__checkpoint:
            x = checkpoint.checkpoint(
                super(BasicNN, self).__call__, x, use_reentrant=False
            )
            _check_first = True
            return x
        else:
            # 启用普通的调用函数
            return super(BasicNN, self).__call__(x)
