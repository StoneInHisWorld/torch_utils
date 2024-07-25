import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils import checkpoint

import utils.func.torch_tools as ttools
from utils.func.torch_tools import sample_wise_ls_fn


class BasicNN(nn.Sequential):
    required_shape = (-1,)

    def __init__(self, *args, **kwargs) -> None:
        """基本神经网络。
        提供神经网络的基本功能，包括权重初始化，训练以及测试。
        :param args: 需要添加的网络层
        :param kwargs: 可选提供的参数。
        -- device: 网络所处设备。
        -- init_meth: 网络初始化方法。
        -- init_kwargs: 网络初始化方法所用参数。
        -- with_checkpoint: 是否使用检查点机制。
        """
        # 设置默认值
        init_meth = 'zero' if 'init_meth' not in kwargs.keys() else kwargs['init_meth']
        device = torch.device('cpu') if 'device' not in kwargs.keys() else kwargs['device']
        with_checkpoint = False if 'with_checkpoint' not in kwargs.keys() else kwargs['with_checkpoint']
        init_kwargs = {} if 'init_kwargs' not in kwargs.keys() else kwargs['init_kwargs']
        # is_train = False if 'is_train' not in kwargs.keys() else kwargs['is_train']
        # 初始化各模块
        super(BasicNN, self).__init__(*args)
        self._init_submodules(init_meth, **init_kwargs)
        self.optimizer_s = None
        self.scheduler_s = None
        self.ls_fn_s = None
        # self._gradient_clipping = None
        self.ready = False
        self.apply(lambda m: m.to(device))

        self._device = device
        if with_checkpoint:
            warnings.warn('使用“检查点机制”虽然会减少前向传播的内存使用，但是会大大增加反向传播的计算量！')
        self.__checkpoint = with_checkpoint

    def prepare_training(
            self,
            o_args=None, l_args=None, ls_args=None,
    ):
        """进行训练准备
        :param o_args: 优化器参数。请注意第一项需为优化器类型字符串，其后为每一个优化器的kwargs。
        :param l_args: 学习率规划器器参数。请注意第一项需为学习率规划器类型字符串，其后为每一个学习率规划器的kwargs。
        :param ls_args:
        :return:
        """
        # 如果不指定，则需要设定默认值
        if o_args is None:
            o_args = []
        if l_args is None:
            l_args = []
        if ls_args is None:
            ls_args = []
        if len(o_args) == 0:
            o_args = [('adam', {}), ]
        self.optimizer_s, self.lr_names = self._get_optimizer(*o_args)
        self.scheduler_s = self._get_lr_scheduler(*l_args)
        # 如果不指定，则需要设定默认值
        if len(ls_args) == 0:
            ls_args = ('mse', {})
        self.ls_fn_s, self.loss_names, self.test_ls_names = self._get_ls_fn(*ls_args)
        try:
            # 设置梯度裁剪方法
            self._gradient_clipping = self._gradient_clipping
        except AttributeError:
            self._gradient_clipping = None
        self.ready = True
        return self.optimizer_s, self.scheduler_s, self.ls_fn_s

    def _get_optimizer(self, *args) -> torch.optim.Optimizer or List[torch.optim.Optimizer]:
        """获取网络优化器。
        此方法会在prepare_training()中被调用。
        :param optim_str_s: 优化器类型字符串序列
        :param args: 每个优化器对应关键词参数
        :return: 优化器序列
        """
        optimizer_s, lr_names = [], []
        for i, (type_s, kwargs) in enumerate(args):
            optimizer_s.append(ttools.get_optimizer(self, type_s, **kwargs))
            lr_names.append(f'LR_{i}')
        return optimizer_s, lr_names

    def _get_lr_scheduler(self, optimizer_s, *args):
        """为优化器定制规划器。
        :param args: 多个二元组，元组格式为（规划器类型字符串，本规划器关键词参数）
        :return: 规划器序列。
        """
        return [
            ttools.get_lr_scheduler(optim, ss, **kwargs)
            for optim, (ss, kwargs) in zip(self.optimizer_s, args)
        ]

    def _get_ls_fn(self, *args):
        """获取网络的损失函数序列，并设置网络的损失名称、测试损失名称
        :param args: 多个二元组，元组格式为（损失函数类型字符串，本损失函数关键词参数），若不指定关键词参数，请用空字典。
        :return: 损失函数序列
        """
        ls_fn_s, loss_names = [], []
        for (type_s, kwargs) in args:
            # 根据参数获取损失函数
            loss_names.append(type_s.upper())
            size_averaged = kwargs.pop('size_averaged', True)
            unwrapped_fn = ttools.get_ls_fn(type_s, **kwargs)
            ls_fn_s.append(
                unwrapped_fn if size_averaged else
                lambda x, y: sample_wise_ls_fn(x, y, unwrapped_fn)
            )
        test_ls_names = loss_names
        return ls_fn_s, loss_names, test_ls_names

    @property
    def device(self):
        return self._device

    def _get_comment(self, inputs, predictions, labels, metrics, mfn_name_s, losses):
        size_averaged_msg = ''
        for i, name in enumerate(mfn_name_s):
            metric = metrics[:, i].mean()
            size_averaged_msg += f'{name} = {float(metric): .4f}, '
        for i, name in enumerate(self.test_ls_names):
            ls = losses[:, i].mean()
            size_averaged_msg += f'{name} = {float(ls): .4f}, '
        comments = []
        for input, pred, lb, metric_s, ls_es in zip(inputs, predictions, labels, metrics, losses):
            comments.append(self._comment_impl(
                input, pred, lb, metric_s, mfn_name_s, ls_es
            ) + '\nSIZE_AVERAGED:\n' + size_averaged_msg)
        return comments

    def _comment_impl(self, input, pred, lb, metric_s, mfn_name_s, ls_es):
        comment = ''
        for metric, name in zip(metric_s, mfn_name_s):
            comment += f'{name} = {float(metric): .4f}, '
        for ls, name in zip(ls_es, self.test_ls_names):
            comment += f'{name} = {float(ls): .4f}, '
        return comment

    def _init_submodules(self, init_str, **kwargs):
        """初始化各模块参数。
        该方法会使用init_str所指初始化方法初始化所用层，若要定制化初始模块，请重载本函数。
        :param init_str: 初始化方法类型
        :param kwargs: 初始化方法参数
        :return:
        """
        init_str = ttools.init_wb(init_str)
        if init_str is not None:
            self.apply(init_str)
        else:
            try:
                where = kwargs['where']
                if where.endswith('.ptsd'):
                    paras = torch.load(where) if torch.cuda.is_available() else \
                        torch.load(where, map_location=torch.device('cpu'))
                    self.load_state_dict(paras)
                elif where.endswith('.ptm'):
                    raise NotImplementedError('针对预训练好的网络，请使用如下方法获取`net = torch.load("../xx.ptm")`')
            except IndexError:
                raise ValueError('选择预训练好的参数初始化网络，需要使用where关键词提供参数或者模型的路径！')

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
                for optim in self._optimizer_s:
                    optim.zero_grad()
                result = self._forward_impl(X, y)
                self._backward_impl(*result[1])
                if self._gradient_clipping is not None:
                    self._gradient_clipping()
                for optim in self._optimizer_s:
                    optim.step()
            assert len(result) == 2, f'前反向传播需要返回元组（预测值，损失值集合），但实现返回的值为{result}'
            assert len(result[1]) == len(self.loss_names), \
                f'前向传播返回的损失值数量{result[1]}与指定的损失名称数量{len(self.loss_names)}不匹配。'
        else:
            with torch.no_grad():
                result = self._forward_impl(X, y)
            assert len(result) == 2, f'前反向传播需要返回元组（预测值，损失值集合），但实现返回的值为{result}'
            assert len(result[1]) == len(self.test_ls_names), \
                f'前向传播返回的损失值数量{result[1]}与指定的损失名称数量{len(self.loss_names)}不匹配。'
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
        return pred, [ls_fn(pred, y) for ls_fn in self.ls_fn_s]

    def _backward_impl(self, *ls):
        ls[0].backward()

    def get_clone_function(self):
        parameter_group = {name: param for name, param in self._construction_parameters.items()}
        args_group = [
            k for k, _ in filter(
                lambda p: p[1].kind == p[1].POSITIONAL_OR_KEYWORD or
                          p[1].kind == p[1].POSITIONAL_ONLY or
                          p[1].kind == p[1].VAR_POSITIONAL,
                parameter_group.items()
            )
        ]
        kwargs_group = [
            k for k, _ in filter(
                lambda p: p[1].kind == p[1].KEYWORD_ONLY or p[1].kind == p[1].VAR_KEYWORD,
                parameter_group.items()
            )
        ]
        kwargs = {}
        for k in kwargs_group:
            kwargs.update(self._construction_variables[k])
        return [
            self._construction_variables[k] for k in args_group
        ], kwargs

    def __str__(self):
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self._device)

    def __call__(self, x):
        if self.__checkpoint:
            _check_first = False
            for m in self:
                can_check = _check_first and type(m) != nn.Dropout and type(m) != nn.BatchNorm2d
                x = checkpoint.checkpoint(m, x) if can_check else m(x)
                _check_first = True
            return x
        else:
            return super(BasicNN, self).__call__(x)
