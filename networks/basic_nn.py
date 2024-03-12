import os.path
import warnings
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from networks.trainer import Trainer
from torch.nn import Module
from torch.utils import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.func.torch_tools as ttools
from data_related.dataloader import LazyDataLoader
from utils.accumulator import Accumulator


class BasicNN(nn.Sequential):
    required_shape = (-1,)

    def __init__(self, *args: Module, **kwargs) -> None:
        """基本神经网络。
        提供神经网络的基本功能，包括权重初始化，训练以及测试。
        :param args: 需要添加的网络层
        :param kwargs: 可选提供的参数。
        -- device: 网络所处设备。
        -- init_meth: 网络初始化方法。
        -- init_args: 网络初始化方法所用参数。
        -- with_checkpoint: 是否使用检查点机制。
        """
        # 设置默认值
        init_meth = 'zero' if 'init_meth' not in kwargs.keys() else kwargs['init_meth']
        device = torch.device('cpu') if 'device' not in kwargs.keys() else kwargs['device']
        with_checkpoint = False if 'with_checkpoint' not in kwargs.keys() else kwargs['with_checkpoint']
        init_args = [] if 'init_args' not in kwargs.keys() else kwargs['init_args']
        # 初始化各模块
        super(BasicNN, self).__init__(*args)
        self.__init_submodules(init_meth, *init_args)
        self._optimizer_s = None
        self._scheduler_s = None
        self._ls_fn = None
        self.for_and_backward = None
        self.is_train = False
        self.apply(lambda m: m.to(device))

        self.__device = device
        if with_checkpoint:
            warnings.warn('使用“检查点机制”虽然会减少前向传播的内存使用，但是会大大增加反向传播的计算量！')
        self.__checkpoint = with_checkpoint

    def prepare_training(self,
                         o_args: tuple = ('adam', ()), l_args: tuple = ([], ()),
                         ls_args: tuple = (), ls_kwargs: dict = {}
                         ):
        """进行训练准备
        :param o_args: 优化器参数。请注意第一项需为优化器类型字符串，其后为每一个优化器的kwargs。
        :param l_args: 学习率规划器器参数。请注意第一项需为学习率规划器类型字符串，其后为每一个学习率规划器的kwargs。
        :param ls_args:
        :param ls_kwargs:
        :return:
        """
        o_args = o_args if isinstance(o_args[0], list) else ([o_args[0]], *o_args[1:])
        self._optimizer_s = self._get_optimizer(*o_args)
        l_args = l_args if isinstance(l_args[0], list) else ([l_args[0]], *l_args[1:])
        self._scheduler_s = self._get_lr_scheduler(*l_args)
        # TODO：暂时不支持多损失函数
        self._ls_fn = self._get_ls_fn(*ls_args, **ls_kwargs)
        #
        # def for_and_backward():
        #     def for_and_backward(X, y, backward=True):
        #
        #         if backward:
        #             pred = self(X)
        #             for optim in self.optimizer_s:
        #                 optim.zero_grad()
        #             lo = self.ls_fn(pred, y)
        #             lo.backward()
        #             for optim in self.optimizer_s:
        #                 optim.step()
        #         else:
        #             with torch.no_grad():
        #                 pred = self(X)
        #                 lo = self.ls_fn(pred, y)
        #         return pred, (lo.item(), )

        self.is_train = True

    def _get_optimizer(self, optim_str_s, *args,
                       ) -> torch.optim.Optimizer or List[torch.optim.Optimizer]:
        optimizer_s, self.lr_names = [], []
        i = 0
        for s, kwargs in zip(optim_str_s, args):
            optimizer_s.append(ttools.get_optimizer(self, s, **kwargs))
            self.lr_names.append(f'LR_{i}')
        return optimizer_s

    def _get_lr_scheduler(self, scheduler_str_s=None, *args):
        """为优化器定制规划器
        :param scheduler_str_s:
        :param args:
        :return:
        """
        if scheduler_str_s is None:
            scheduler_str_s = []
        # 如果参数太少，则用空字典补齐
        if len(args) <= len(self._optimizer_s):
            args = (*args, *[{} for _ in range(len(scheduler_str_s) - len(args))])
        return [
            ttools.get_lr_scheduler(optim, ss, **kwargs)
            for ss, optim, kwargs in zip(scheduler_str_s, self._optimizer_s, args)
        ]
        # else:
        #     if kwargs == {}:
        #         kwargs = {'step_size': 1, 'gamma': 1}
        #     return ttools.get_lr_scheduler(self.optimizer_s, scheduler_str_s, **kwargs)

    def _get_ls_fn(self, ls_fn='mse', **kwargs):
        self.loss_names = [ls_fn.upper()]
        ls_fn = ttools.get_ls_fn(ls_fn, **kwargs)
        return lambda X, y: ls_fn(X, y)

    # def train_(self,
    #            data_iter, optimizers, acc_fn,
    #            n_epochs=10, ls_fn: nn.Module = nn.L1Loss(), lr_schedulers=None,
    #            valid_iter=None, k=1, n_workers=1, hook=None
    #            ):
    #     """神经网络训练函数
    #     调用Trainer进行训练。
    #     :param data_iter: 训练数据供给迭代器。
    #     :param optimizers: 网络参数优化器。可以通过list传入多个优化器。
    #     :param acc_fn: 准确率计算函数。签名需为：acc_fn(predict: tensor, labels: tensor, size_averaged: bool = True) -> ls: tensor or float
    #     :param n_epochs: 迭代世代数。
    #     :param ls_fn: 训练损失函数。签名需为：ls_fn(predict: tensor, labels: tensor) -> ls: tensor，其中不允许有销毁梯度的操作。
    #     :param lr_schedulers: 学习率规划器，用于动态改变学习率。若不指定，则会使用固定学习率规划器。
    #     :param valid_iter: 验证数据供给迭代器。
    #     :param hook: 是否使用hook机制跟踪梯度变化。可选填入[None, 'mute', 'full']
    #     :param n_workers: 进行训练的处理机数量。
    #     :param k: 进行训练的k折数，指定为1则不进行k-折训练，否则进行k-折训练，并且data_iter为k-折训练数据供给器，valid_iter会被忽略.
    #     :return: 训练数据记录`History`对象
    #     """
    #     assert self.is_train, '在训练之前，请先调用prepare_training()！'
    #     if hook is None:
    #         with_hook, hook_mute = False, False
    #     elif hook == 'mute':
    #         with_hook, hook_mute = True, True
    #     else:
    #         with_hook, hook_mute = True, False
    #     if lr_schedulers is None:
    #         if isinstance(optimizers, list):
    #             lr_schedulers = [torch.optim.lr_scheduler.StepLR(optim, 1, 1) for optim in optimizers]
    #         else:
    #             lr_schedulers = torch.optim.lr_scheduler.StepLR(optimizers, 1, 1)
    #     with Trainer(self,
    #                  data_iter, optimizers, lr_schedulers, acc_fn, n_epochs, ls_fn,
    #                  with_hook, hook_mute) as trainer:
    #         if k > 1:
    #             # 是否进行k-折训练
    #             history = trainer.train_with_k_fold(data_iter, k, n_workers)
    #         elif n_workers <= 1:
    #             # 判断是否要多线程
    #             if valid_iter:
    #                 history = trainer.train_and_valid(valid_iter)
    #             else:
    #                 history = trainer.train()
    #         else:
    #             history = trainer.train_with_threads(valid_iter)
    #     self.is_train = False
    #     return history

    def train_(self,
               data_iter, criterion_a,
               n_epochs=10, valid_iter=None, k=1, n_workers=1, hook=None
               ):
        """神经网络训练函数
        调用Trainer进行训练。
        :param data_iter: 训练数据供给迭代器。
        :param optimizers: 网络参数优化器。可以通过list传入多个优化器。
        :param criterion_a: 准确率计算函数。签名需为：acc_fn(predict: tensor, labels: tensor, size_averaged: bool = True) -> ls: tensor or float
        :param n_epochs: 迭代世代数。
        :param ls_fn: 训练损失函数。签名需为：ls_fn(predict: tensor, labels: tensor) -> ls: tensor，其中不允许有销毁梯度的操作。
        :param lr_schedulers: 学习率规划器，用于动态改变学习率。若不指定，则会使用固定学习率规划器。
        :param valid_iter: 验证数据供给迭代器。
        :param hook: 是否使用hook机制跟踪梯度变化。可选填入[None, 'mute', 'full']
        :param n_workers: 进行训练的处理机数量。
        :param k: 进行训练的k折数，指定为1则不进行k-折训练，否则进行k-折训练，并且data_iter为k-折训练数据供给器，valid_iter会被忽略.
        :return: 训练数据记录`History`对象
        """
        assert self.is_train, '在训练之前，请先调用prepare_training()！'
        if hook is None:
            with_hook, hook_mute = False, False
        elif hook == 'mute':
            with_hook, hook_mute = True, True
        else:
            with_hook, hook_mute = True, False
        with Trainer(self,
                     data_iter, self._optimizer_s, self._scheduler_s, criterion_a,
                     self.for_and_backward, n_epochs, self._ls_fn, with_hook, hook_mute
                     ) as trainer:
            if k > 1:
                # 是否进行k-折训练
                history = trainer.train_with_k_fold(data_iter, k, n_workers)
            elif n_workers <= 1:
                # 判断是否要多线程
                if valid_iter:
                    history = trainer.train_and_valid(valid_iter)
                else:
                    history = trainer.train()
            else:
                history = trainer.train_with_threads(valid_iter)
        # 清楚训练痕迹
        del self._optimizer_s, self._scheduler_s, self._ls_fn
        self.is_train = False
        return history

    @torch.no_grad()
    def test_(self, test_iter,
              criterion_a: Callable[[torch.Tensor, torch.Tensor], float or torch.Tensor],
              is_valid: bool = False, ls_fn=None, **ls_fn_kwargs
              ) -> [float, float]:
        """测试方法。
        取出迭代器中的下一batch数据，进行预测后计算准确度和损失
        :param test_iter: 测试数据迭代器
        :param criterion_a: 计算准确度所使用的函数，该函数需要求出整个batch的准确率之和。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param l_names: 计算损失所使用的函数，该函数需要求出整个batch的损失平均值。签名需为：loss(Y_HAT, Y) -> float or torch.Tensor
        :return: 测试准确率，测试损失
        """
        self.eval()
        criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
        if not is_valid:
            # 如果是进行测试，则需要先初始化损失函数。
            if ls_fn is None:
                ls_fn = 'mse'
            self._ls_fn = self._get_ls_fn(ls_fn, **ls_fn_kwargs)
            test_iter = tqdm(test_iter, unit='批', position=0, desc=f'测试中……', mininterval=1)
        # 要统计的数据种类数目
        l_names = self.loss_names if isinstance(self.loss_names, list) else [self.loss_names]
        metric = Accumulator(len(criterion_a) + len(l_names) + 1)
        # 计算准确率和损失值
        for features, labels in test_iter:
            preds, ls_es = self.forward_backward(features, labels, False)
            metric.add(
                *[criterion(preds, labels) for criterion in criterion_a],
                *[ls * len(features) for ls in ls_es], len(features)
            )
        # 生成测试日志
        log = {}
        if is_valid:
            prefix = 'valid_'
        else:
            prefix = 'test_'
            del self._ls_fn
        i = 0
        for i, computer in enumerate(criterion_a):
            try:
                log[prefix + computer.__name__] = metric[i] / metric[-1]
            except AttributeError:
                log[prefix + computer.__class__.__name__] = metric[i] / metric[-1]
        for j, loss_name in enumerate(l_names):
            log[prefix + loss_name] = metric[i + j] / metric[-1]
        return log
        # self.eval()
        # metric = Accumulator(3)
        # for features, labels in test_iter:
        #     preds = self(features)
        #     metric.add(acc_fn(preds, labels), ls_fn(preds, labels) * len(features), len(features))
        # return metric[0] / metric[2], metric[1] / metric[2]

    @torch.no_grad()
    def predict_(self, data_iter: DataLoader or LazyDataLoader,
                 acc_fn: Callable,
                 ls_fn: Callable[[torch.Tensor, torch.Tensor], float or torch.Tensor] = nn.L1Loss,
                 unwrap_fn: Callable[
                     [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = None
                 ) -> torch.Tensor:
        """预测方法。
        对于数据迭代器中的每一batch数据，保存输入数据、预测数据、标签集、准确率、损失值数据，并打包返回。
        :param data_iter: 预测数据迭代器。
        :param acc_fn: 计算准确度所使用的函数，该函数需要求出整个batch的准确率之和。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param ls_fn: 计算损失所使用的函数，该函数需要求出整个batch的损失平均值。签名需为：loss(Y_HAT, Y) -> float or torch.Tensor
        :param unwrap_fn: 对所有数据进行打包的方法。如不指定，则直接返回预测数据。签名需为：unwrap_fn(inputs, predictions, labels, acc_s, loss_es) -> Any
        :return: 打包好的数据集
        """
        self.eval()
        if unwrap_fn is not None:
            # 将本次预测所产生的全部数据打包并返回。
            inputs, predictions, labels, acc_s, loss_es = [], [], [], [], []
            with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
                # 对每个批次进行预测，并进行acc和loss的计算
                for fe_batch, lb_batch in data_iter:
                    inputs.append(fe_batch)
                    pre_batch = self(fe_batch)
                    predictions.append(pre_batch)
                    labels.append(lb_batch)
                    # for pre, lb in zip(pre_batch, lb_batch):
                    #     acc_s.append(
                    #         acc_fn(pre.reshape(1, *pre.shape), lb.reshape(1, *pre.shape))
                    #     )
                    #     loss_es.append(ls_fn(pre, lb))
                    acc_s += acc_fn(pre_batch, lb_batch, size_average=False)
                    keeping_dims = list(range(len(fe_batch.shape)))[1:]
                    loss_es += ls_fn(pre_batch, lb_batch).mean(dim=keeping_dims)
                data_iter.set_description('正对结果进行解包……')
            inputs = torch.cat(inputs, dim=0)
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)
            acc_s = torch.tensor(acc_s)
            loss_es = torch.tensor(loss_es)
            predictions = unwrap_fn(inputs, predictions, labels, acc_s, loss_es)
        else:
            # 如果不需要打包数据，则直接返回预测数据集
            predictions = []
            with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
                for fe_batch, lb_batch in data_iter:
                    predictions.append(self(fe_batch))
            predictions = torch.cat(predictions, dim=0)
        return predictions

    @property
    def device(self):
        return self.__device

    def load_state_dict_(self, where: str):
        assert os.path.exists(where), f'目录{where}无效！'
        assert where.endswith('.ptsd'), f'该文件{where}并非网络参数文件！'
        paras = torch.load(where)
        self.load_state_dict(paras)

    def __init_submodules(self, init_meth, *args):
        init_meth = ttools.init_wb(init_meth)
        if init_meth is not None:
            self.apply(init_meth)
        else:
            try:
                where = args[0]
                if where.endswith('.ptsd'):
                    paras = torch.load(where)
                    self.load_state_dict(paras)
                elif where.endswith('.ptm'):
                    raise NotImplementedError('针对预训练好的网络，请使用如下方法获取`net = torch.load("../xx.ptm")`')
            except IndexError:
                raise ValueError('选择预训练好的参数初始化网络，需要在初始化方法的第一参数提供参数或者模型的路径！')
            except FileNotFoundError:
                raise FileNotFoundError(f'找不到{where}！')

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
                result = self.__forward_impl(X, y)
                self.__backward_impl(*result[1])
        else:
            with torch.no_grad():
                result = self.__forward_impl(X, y)
        assert len(result) == 2, f'前反向传播需要返回元组（预测值，损失值集合），但实现返回的值为{result}'
        assert len(result[1]) == len(self.loss_names), f'前向传播返回的损失值数量{result[1]}与指定的损失名称数量{len(self.loss_names)}不匹配。'
        return result

    def __forward_impl(self, X, y) -> Tuple[torch.Tensor, Tuple]:
        """前向传播实现。
        进行前向传播后，根据self._ls_fn()计算损失值，并返回。
        若要更改optimizer.zero_grad()以及backward()的顺序，请直接重载forward_backward()！
        :param X: 特征集
        :param y: 标签集
        :return: （预测值， （损失值集合））
        """
        pred = self(X)
        return pred, (self._ls_fn(pred, y),)

    def __backward_impl(self, ls):
        ls.backward()
        for optim in self._optimizer_s:
            optim.zero_grad()
            optim.step()

    def __str__(self):
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device)

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
