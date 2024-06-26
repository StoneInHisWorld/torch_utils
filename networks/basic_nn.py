import warnings
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.func.torch_tools as ttools
from data_related.dataloader import LazyDataLoader
from networks.trainer import Trainer
from utils.accumulator import Accumulator
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
        is_train = True if 'is_train' not in kwargs.keys() else kwargs['is_train']
        # 初始化各模块
        super(BasicNN, self).__init__(*args)
        self._init_submodules(init_meth, **init_kwargs)
        self._optimizer_s = None
        self._scheduler_s = None
        self._ls_fn_s = None
        # self._gradient_clipping = None
        self.is_train = is_train
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
            o_args = ('adam', {})
        self._optimizer_s, self.lr_names = self._get_optimizer(*o_args)
        self._scheduler_s = self._get_lr_scheduler(*l_args)
        # 如果不指定，则需要设定默认值
        if len(ls_args) == 0:
            ls_args = ('mse', {})
        # TODO:测试到这里
        self._ls_fn_s, self.loss_names, self.test_ls_names = self._get_ls_fn(*ls_args)
        try:
            # 设置梯度裁剪方法
            self._gradient_clipping = self._gradient_clipping
        except AttributeError:
            self._gradient_clipping = None
        self.is_train = True

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

    def _get_lr_scheduler(self, *args):
        """为优化器定制规划器。
        :param args: 多个二元组，元组格式为（规划器类型字符串，本规划器关键词参数）
        :return: 规划器序列。
        """
        return [
            ttools.get_lr_scheduler(optim, ss, **kwargs)
            for optim, (ss, kwargs) in zip(self._optimizer_s, args)
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

    # def train_(self,
    #            data_iter, criterion_a,
    #            n_epochs=10, valid_iter=None, k=1, n_workers=1, hook=None
    #            ):
    #     """神经网络训练函数
    #     调用Trainer进行训练。
    #     :param data_iter: 训练数据供给迭代器。
    #     :param optimizers: 网络参数优化器。可以通过list传入多个优化器。
    #     :param criterion_a: 准确率计算函数。签名需为：acc_fn(predict: tensor, labels: tensor, size_averaged: bool = True) -> ls: tensor or float
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
    #     with Trainer(self,
    #                  data_iter, self._optimizer_s, self._scheduler_s, criterion_a,
    #                  n_epochs, self._ls_fn_s, with_hook, hook_mute
    #                  ) as trainer:
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
    #             history = trainer.train_with_threads(valid_iter, n_workers)
    #     # 清楚训练痕迹
    #     del self._optimizer_s, self.lr_names, self._scheduler_s, \
    #         self._ls_fn_s, self.loss_names, self.test_ls_names
    #     self.is_train = False
    #     return history

    def train_(self,
               data_iter, criterion_a,
               n_epochs=10, k=1, n_workers=1, hook=None
               ):
        """神经网络训练函数
        调用Trainer进行训练。
        :param data_iter: 训练数据供给迭代器。
        :param optimizers: 网络参数优化器。可以通过list传入多个优化器。
        :param criterion_a: 准确率计算函数。签名需为：acc_fn(predict: tensor, labels: tensor, size_averaged: bool = True) -> ls: tensor or float
        :param n_epochs: 迭代世代数。
        :param ls_fn: 训练损失函数。签名需为：ls_fn(predict: tensor, labels: tensor) -> ls: tensor，其中不允许有销毁梯度的操作。
        :param lr_schedulers: 学习率规划器，用于动态改变学习率。若不指定，则会使用固定学习率规划器。
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
                     self._optimizer_s, self._scheduler_s, criterion_a,
                     n_epochs, self._ls_fn_s, with_hook, hook_mute
                     ) as trainer:
            if k > 1:
                # 是否进行k-折训练
                history = trainer.train_with_k_fold(data_iter, k, n_workers)
            else:
                # 提取训练迭代器和验证迭代器
                data_iter = list(data_iter)
                if len(data_iter) == 2:
                    train_iter, valid_iter = [it[0] for it in data_iter]
                elif len(data_iter) == 1:
                    train_iter, valid_iter = data_iter[0][0], None
                else:
                    raise ValueError(f"无法识别的数据迭代器，其提供的长度为{len(data_iter)}")
                # 判断是否要进行多线程训练
                if n_workers <= 1:
                    # 不启用多线程训练
                    if valid_iter is not None:
                        history = trainer.train_and_valid(train_iter, valid_iter)
                    else:
                        history = trainer.train(train_iter)
                else:
                    # 启用多线程训练
                    history = trainer.train_with_threads(train_iter, valid_iter, None, n_workers)
        # 清楚训练痕迹
        del self._optimizer_s, self.lr_names, self._scheduler_s, \
            self._ls_fn_s, self.loss_names, self.test_ls_names
        self.is_train = False
        return history

    @torch.no_grad()
    def test_(self, test_iter,
              criterion_a: List[Callable],
              pbar=None, ls_fn_args=None
              ) -> [float, float]:
        """测试方法。
        取出迭代器中的下一batch数据，进行预测后计算准确度和损失
        :param test_iter: 测试数据迭代器
        :param criterion_a: 计算准确度所使用的函数，该函数需要求出整个batch的准确率之和。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param l_names: 计算损失所使用的函数，该函数需要求出整个batch的损失平均值。签名需为：loss(Y_HAT, Y) -> float or torch.Tensor
        :return: 测试准确率，测试损失
        """
        self.eval()
        non_blocking = self.device.type == 'cuda' and test_iter.pin_memory
        # 要统计的数据种类数目
        criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
        is_test = pbar is None
        if is_test:
            # 如果不指定进度条，则是进行测试，需要先初始化损失函数。
            if ls_fn_args is None:
                ls_fn_args = [('mse', {})]
            self._ls_fn_s, _, self.test_ls_names = self._get_ls_fn(*ls_fn_args)
            pbar = tqdm(test_iter, unit='批', position=0, desc=f'测试中……', mininterval=1)
        else:
            pbar.set_description('验证中……')
        # 要统计的数据种类数目
        # l_names = self.test_ls_names if isinstance(self.test_ls_names, list) else [self.test_ls_names]
        l_names = self.test_ls_names
        metric = Accumulator(len(criterion_a) + len(l_names) + 1)
        # 计算准确率和损失值
        for features, labels in test_iter:
            features, labels = (
                features.to(self.device, non_blocking=non_blocking),
                labels.to(self.device, non_blocking=non_blocking)
            )
            preds, ls_es = self.forward_backward(features, labels, False)
            metric.add(
                *[criterion(preds, labels) for criterion in criterion_a],
                *[ls * len(features) for ls in ls_es],
                len(features)
            )
            pbar.update(1)
        # 生成测试日志
        log = {}
        if is_test:
            prefix = 'test_'
            del self._ls_fn_s
        else:
            prefix = 'valid_'
        i = 0
        for i, computer in enumerate(criterion_a):
            try:
                log[prefix + computer.__name__] = metric[i] / metric[-1]
            except AttributeError:
                log[prefix + computer.__class__.__name__] = metric[i] / metric[-1]
        i += 1
        for j, loss_name in enumerate(l_names):
            log[prefix + loss_name] = metric[i + j] / metric[-1]
        return log

    @torch.no_grad()
    def predict_(self, data_iter: DataLoader or LazyDataLoader,
                 criterion_a: Callable or List[Callable],
                 unwrap_fn: Callable = None,
                 ls_fn_args: Tuple = ()
                 ) -> torch.Tensor:
        """预测方法。
        对于数据迭代器中的每一batch数据，保存输入数据、预测数据、标签集、准确率、损失值数据，并打包返回。
        :param ls_fn_args: 损失函数序列的关键词参数
        :param data_iter: 预测数据迭代器。
        :param criterion_a: 计算准确度所使用的函数序列，该函数需要求出每个样本的准确率。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param unwrap_fn: 对所有数据进行打包的方法。如不指定，则直接返回预测数据。签名需为：unwrap_fn(inputs, predictions, labels, metrics, losses) -> Any
        :return: 打包好的数据集
        """
        self.eval()
        criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
        # 预处理损失函数参数
        for (_, kwargs) in ls_fn_args:
            kwargs['size_averaged'] = False
        self._ls_fn_s, _, self.test_ls_names = self._get_ls_fn(*ls_fn_args)
        if unwrap_fn is not None:
            # 将本次预测所产生的全部数据打包并返回。
            inputs, predictions, labels, metrics, losses = [], [], [], [], []
            with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
                # 对每个批次进行预测，并进行acc和loss的计算
                for fe_batch, lb_batch in data_iter:
                    result = self.forward_backward(fe_batch, lb_batch, False)
                    pre_batch, ls_es = result
                    inputs.append(fe_batch)
                    predictions.append(pre_batch)
                    labels.append(lb_batch)
                    metrics_to_be_appended = [
                        criterion(pre_batch, lb_batch, size_averaged=False)
                        for criterion in criterion_a
                    ]
                    metrics.append(torch.vstack(metrics_to_be_appended).T)
                    losses.append(torch.vstack(ls_es).T)
                data_iter.set_description('结果计算完成')
            # 将所有批次的数据堆叠在一起
            inputs = torch.cat(inputs, dim=0)
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)
            metrics = torch.cat(metrics, dim=0)
            losses = torch.cat(losses, dim=0)
            # 获取输出结果需要的注解
            comments = self._get_comment(
                inputs, predictions, labels,
                metrics, [criterion.__name__ for criterion in criterion_a],
                losses
            )
            # 将注解与所有数据打包，输出
            predictions = unwrap_fn(
                inputs, predictions, labels, metrics, losses, comments
            )
        else:
            # TODO: Untested!
            # 如果不需要打包数据，则直接返回预测数据集
            predictions = []
            with tqdm(data_iter, unit='批', position=0, desc=f'正在计算结果……', mininterval=1) as data_iter:
                for fe_batch, lb_batch in data_iter:
                    predictions.append(self(fe_batch))
            predictions = torch.cat(predictions, dim=0)
        return predictions

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
            assert len(result[1]) == len(
                self.loss_names), f'前向传播返回的损失值数量{result[1]}与指定的损失名称数量{len(self.loss_names)}不匹配。'
        else:
            with torch.no_grad():
                result = self._forward_impl(X, y)
            assert len(result) == 2, f'前反向传播需要返回元组（预测值，损失值集合），但实现返回的值为{result}'
            assert len(result[1]) == len(
                self.test_ls_names), f'前向传播返回的损失值数量{result[1]}与指定的损失名称数量{len(self.loss_names)}不匹配。'
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
        return pred, [ls_fn(pred, y) for ls_fn in self._ls_fn_s]

    def _backward_impl(self, *ls):
        ls[0].backward()

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
