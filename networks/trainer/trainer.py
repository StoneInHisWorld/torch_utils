import os.path
import time

import dill
import torch
from torch.multiprocessing import Event as PEvent
from torch.multiprocessing import Process
from torch.multiprocessing import SimpleQueue as PQueue
from tqdm import tqdm

from networks import BasicNN
from networks.trainer.__hook_impl import hook
from networks.trainer.__profiler_impl import profiling_impl
from networks.trainer.__subprocess_impl import data_iter_impl, train_impl, eval_impl, vlog_impl, \
    tlog_impl
from utils.accumulator import Accumulator
from networks.decorators import prepare
from utils.func import pytools as ptools
# from utils.func.pytools import get_computer_name, is_multiprocessing
from utils.history import History


class Trainer:
    """神经网络训练器对象，提供所有针对神经网络的操作，包括训练、验证、测试、预测"""

    def __init__(self,
                 module_class, m_init_args, m_init_kwargs, input_size, prepare_args,  # 网络模型相关构造参数
                 criterion_a, runtime_cfg,
                 hps=None,  # 训练、验证、测试依赖参数
                 ):
        """神经网络训练器对象，提供所有针对神经网络的操作，包括训练、验证、测试、预测。
        Trainer类负责进行网络构建以及网络训练方法实现，可以通过Trainer.module获取训练完成或正在训练的网络对象。

        :param module_class: 模型类，用于调用构造函数实现网络构造
        :param m_init_args: 网络构造位置参数
        :param m_init_kwargs: 网络构造关键字参数
        :param prepare_args: 网络训练准备参数，用于BasicNN中的prepare_training()函数
        :param criterion_a: 评价指标计算方法，如需指定多个请传入`list`对象
        :param hps: 超参数组合，用于获取其中的超参数进行训练
        :param runtime_cfg: 动态运行参数，用于获取其中的参数指导训练
        """
        # # TODO: 进行默认值检查和填充
        # if runtime_cfg is None:
        #     runtime_cfg = {'print_net': False, 'with_checkpoint': False}
        if hps is None:
            hps = {}
        assert issubclass(module_class, BasicNN), f'模型类型{module_class}未知，请使用net包下实现的网络模型'
        # 存放网络初始化参数
        self.module_class = module_class
        self.m_init_args = m_init_args
        self.m_init_kwargs = m_init_kwargs
        self.prepare_args = prepare_args
        self.module = None
        # 设置训练、验证、测试依赖参数
        self.input_size = input_size
        self.hps = hps
        self.runtime_cfg = runtime_cfg
        self.criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
        self.pbar = None

    def train(self, data_iter) -> History:
        """训练公共接口。
        拆解数据迭代器，并根据训练器超参数以及动态运行参数判断进行的训练类型，调用相应训练函数。
        训练进度条在此创建。

        :param data_iter: 训练所用数据迭代器
        :return: 训练数据记录对象
        """
        # 提取所需超参数以及动态运行参数
        n_workers = self.runtime_cfg['n_workers']
        n_epochs = self.hps['epochs']
        k = self.hps['k']
        print(f'\r本次训练位于设备{self.runtime_cfg["device"]}上', flush=True)
        # 判断是否是k折训练
        if k > 1:
            pbar_len = k * n_epochs
            train_fn, train_args = self.__train_with_k_fold, (data_iter,)
        else:
            # 提取训练迭代器和验证迭代器
            data_iter = list(data_iter)
            if len(data_iter) == 2:
                train_iter, valid_iter = [it[0] for it in data_iter]
                pbar_len = (len(train_iter) + len(valid_iter)) * n_epochs
            elif len(data_iter) == 1:
                train_iter, valid_iter = data_iter[0][0], None
                pbar_len = len(train_iter) * n_epochs
            else:
                raise ValueError(f"无法识别的数据迭代器，其提供的长度为{len(data_iter)}")
            # 判断是否要进行多线程训练
            if not ptools.is_multiprocessing(n_workers):
                # 不启用多线程训练
                if valid_iter is not None:
                    # 进行训练和验证
                    train_fn, train_args = self.__train_and_valid, (train_iter, valid_iter)
                else:
                    # 进行训练
                    train_fn, train_args = self.__train, (train_iter,)
            else:
                # 启用多进程训练
                train_fn, train_args = self.__train_with_multithreading, (train_iter, valid_iter)
        # 设置进度条
        pbar = tqdm(
            total=pbar_len, unit='批', position=0,
            desc=f'正在进行训练准备……', mininterval=1, ncols=100
        )
        self.pbar = pbar  # 多进程运行需要删除此属性，此举防止pbar被回收
        history = train_fn(*train_args)
        pbar.close()
        return history

    @prepare('predict')
    def predict(self, wrap_fn=None, save_path=None):
        """预测方法。
        对于数据迭代器中的每一batch数据，保存输入数据、预测数据、标签集、准确率、损失值数据，并打包返回。
        :param ls_fn_args: 损失函数序列的关键词参数
        :param data_iter: 预测数据迭代器。
        :param criterion_a: 计算准确度所使用的函数序列，该函数需要求出每个样本的准确率。签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param wrap_fn: 对所有数据进行打包的方法。如不指定，则直接返回预测数据。签名需为：unwrap_fn(inputs, predictions, labels, metrics, losses) -> Any
        :return: 打包好的数据集
        """
        # 提取训练器参数
        net = self.module
        criterion_a = self.criterion_a
        pbar = self.pbar
        # 如果传递了包装方法
        if wrap_fn is not None:
            # 将本次预测所产生的全部数据打包并返回。
            inputs, predictions, labels, metrics, losses = [], [], [], [], []
            # 对每个批次进行预测，并进行评价指标和损失值的计算
            for fe_batch, lb_batch in pbar:
                result = net.forward_backward(fe_batch, lb_batch, False)
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
            pbar.set_description('结果计算完成')
            # 将所有批次的数据堆叠在一起
            inputs = torch.cat(inputs, dim=0)
            predictions = torch.cat(predictions, dim=0)
            labels = torch.cat(labels, dim=0)
            metrics = torch.cat(metrics, dim=0)
            losses = torch.cat(losses, dim=0)
            # 获取输出结果需要的注解
            comments = net.get_comment(
                inputs, predictions, labels,
                metrics, [ptools.get_computer_name(criterion) for criterion in criterion_a],
                losses
            )
            # 将注解与所有数据打包，输出
            predictions = wrap_fn(inputs, predictions, labels, comments)
        else:
            # 如果不需要打包数据，则直接返回预测数据集
            predictions = []
            for fe_batch, lb_batch in pbar:
                predictions.append(net(fe_batch))
            predictions = torch.cat(predictions, dim=0)
        # 如果指定了保存路径，则保存预测结果
        if save_path:
            ptools.check_path(save_path)
            with tqdm(predictions, unit='张', position=0, desc=f"正在保存结果……",
                      mininterval=1, leave=True, ncols=80) as pbar:
                for i, res in enumerate(pbar):
                    res.save(os.path.join(save_path, f'{i}.png'), format='PNG', dpi=(400, 400), compress_level=0)
        return predictions

    @prepare('train')
    @hook()
    def __train(self, train_iter) -> History:
        """简单训练实现。
        从train_iter中每次取出一批次数据进行前反向传播后计算评价指标，记录到History对象中。

        本函数的实际接口为
        def __train(
            module_class: type, __m_init_args: tuple, __m_init_kwargs: dict, prepare_args: tuple,
            train_iter: DataLoader
        ) -> History

        module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
        prepare_args用于给prepare装饰器进行网络训练准备

        :param train_iter: 训练数据迭代器
        :return: 训练数据记录`History`对象
        """
        # 提取训练器参数
        pbar = self.pbar
        net = self.module
        criterion_a = self.criterion_a
        n_epochs = self.hps['epochs']
        optimizer_s = net.optimizer_s
        scheduler_s = net.scheduler_s
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
        # 学习率项
        lr_names = net.lr_names
        history = History(*(criteria_names + loss_names + lr_names))
        for epoch in range(n_epochs):
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
            history.add(
                lr_names, [
                    [param['lr'] for param in optimizer.param_groups]
                    for optimizer in optimizer_s
                ]
            )
            # 记录批次训练损失总和，评价指标，样本数
            metric = Accumulator(len(loss_names + criteria_names) + 1)
            # 训练主循环
            for X, y in train_iter:
                net.train()
                pred, ls_es = net.forward_backward(X, y)
                with torch.no_grad():
                    num_examples = X.shape[0]
                    correct_s = []
                    for criterion in criterion_a:
                        correct = criterion(pred, y)
                        correct_s.append(correct)
                    metric.add(
                        *correct_s, *[ls * num_examples for ls in ls_es],
                        num_examples
                    )
                pbar.update(1)
            for scheduler in scheduler_s:
                scheduler.step()
            history.add(
                criteria_names + loss_names,
                [metric[i] / metric[-1] for i in range(len(metric) - 1)]
            )
        return history

    @prepare('train')
    @hook()
    def __train_and_valid(self, train_iter, valid_iter) -> History:
        """训练和验证实现函数。
        从train_iter中每次取出一批次数据进行前反向传播后计算评价指标获得训练日志，随后调用__valid()函数进行验证获得验证日志，
        最后将两者记录到History对象中。

        本函数的实际签名为：
        def __train_and_valid(
            module_class: type, __m_init_args: tuple, __m_init_kwargs: dict, prepare_args: tuple,
            train_iter: DataLoader, valid_iter: DataLoader,
        ) -> History
        module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
        prepare_args用于给prepare装饰器进行网络训练准备

        :param train_iter: 训练数据供给迭代器
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        # 提取训练器参数
        pbar = self.pbar
        net = self.module
        criterion_a = self.criterion_a
        n_epochs = self.hps['epochs']
        optimizer_s = net.optimizer_s
        scheduler_s = net.scheduler_s
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [
            f'train_{ptools.get_computer_name(criterion)}' for criterion in criterion_a
        ]
        # 学习率项
        lr_names = net.lr_names
        history = History(*(criteria_names + loss_names + lr_names))
        # 世代迭代主循环

        for epoch in range(n_epochs):
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
            history.add(
                lr_names, [
                    [param['lr'] for param in optimizer.param_groups]
                    for optimizer in optimizer_s
                ]
            )
            # 记录批次训练损失总和，评价指标，样本数
            metric = Accumulator(len(loss_names + criteria_names) + 1)
            # 训练主循环
            for X, y in train_iter:
                net.train()
                pred, ls_es = net.forward_backward(X, y)
                with torch.no_grad():
                    num_examples = X.shape[0]
                    correct_s = []
                    for criterion in criterion_a:
                        correct = criterion(pred, y)
                        correct_s.append(correct)
                    metric.add(
                        *correct_s, *[ls * num_examples for ls in ls_es],
                        num_examples
                    )
                pbar.update(1)
            # 进行学习率更新
            for scheduler in scheduler_s:
                scheduler.step()
            valid_log = self.__valid(valid_iter)
            # 记录训练数据
            history.add(
                criteria_names + loss_names + list(valid_log.keys()),
                [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
                list(valid_log.values())
            )
        return history

    @prepare('train')
    def __train_with_k_fold(self, train_loaders_iter) -> History:
        """k-折训练实现

        拆解数据加载器供给器为k折，每折调用__train_and_valid()函数进行训练，获取训练日志后整合成k折训练日志。

        本函数的实际签名为：
        def __train_with_k_fold(
            module_class: type, __m_init_args: tuple, __m_init_kwargs: dict, prepare_args: tuple,
            train_loaders_iter: Generator[DataLoader]
        ) -> History

        module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
        prepare_args用于给prepare装饰器进行网络训练准备

        :param train_loaders_iter: 数据加载器供给器，提供k折验证的每一次训练所需训练集加载器、验证集加载器
        :return: k折训练日志
        """
        # 提取训练器参数
        n_workers = self.runtime_cfg['n_workers']
        n_epochs = self.hps['epochs']
        k = self.hps['k']
        pbar = self.pbar
        # 根据训练器参数调用相应的训练函数
        prepare.k_fold = True
        k_fold_history = None
        for i, (train_iter, valid_iter) in enumerate(train_loaders_iter):
            pbar.set_description(f'\r训练折{i + 1}……')
            # 计算训练批次数
            pbar.total = k * n_epochs * (len(train_iter) + len(valid_iter))
            if ptools.is_multiprocessing(n_workers):
                history = self.__train_with_multithreading(train_iter, valid_iter)
            else:
                history = self.__train_and_valid(train_iter, valid_iter)
            if k_fold_history is None:
                k_fold_history = history
            else:
                k_fold_history += history
            pbar.set_description(f'\r折{i + 1}训练完毕')
        # 去掉prepare中的k_fold标记以提示prepare消除训练痕迹
        prepare.k_fold = False
        return k_fold_history

    @prepare('test')
    def test(self, test_iter) -> dict:
        """测试实现。
        每次取出测试数据供给器中的下一批次数据进行前向传播，计算评价指标和损失，记录到日志中。

        :param test_iter: 测试数据迭代器
        :return: 测试记录
        """
        # 提取训练器参数
        net = self.module
        criterion_a = self.criterion_a
        pbar = tqdm(test_iter, unit='批', position=0, desc=f'测试中……', mininterval=1, ncols=100)
        # 要统计的数据种类数目
        l_names = net.test_ls_names
        metric = Accumulator(len(criterion_a) + len(l_names) + 1)
        # 计算准确率和损失值
        for features, labels in test_iter:
            preds, ls_es = net.forward_backward(features, labels, False)
            metric.add(
                *[criterion(preds, labels) for criterion in criterion_a],
                *[ls * len(features) for ls in ls_es],
                len(features)
            )
            pbar.update(1)
        # 生成测试日志
        log = {}
        i = 0
        for i, computer in enumerate(criterion_a):
            log['test_' + ptools.get_computer_name(computer)] = metric[i] / metric[-1]
        i += 1
        for j, loss_name in enumerate(l_names):
            log['test_' + loss_name] = metric[i + j] / metric[-1]
        return log

    @prepare('valid')
    def __valid(self, valid_iter) -> [float, float]:
        """验证函数实现
        每次取出验证数据供给器中的下一批次数据进行前向传播，之后计算评价指标和损失，生成验证日志。

        :param valid_iter: 验证数据供给器
        :return: 验证记录
        """
        # 提取出验证所需参数
        net = self.module
        criterion_a = self.criterion_a
        pbar = self.pbar
        # 要统计的数据种类数目
        l_names = net.test_ls_names
        metric = Accumulator(len(criterion_a) + len(l_names) + 1)
        # 计算准确率和损失值
        for features, labels in valid_iter:
            preds, ls_es = net.forward_backward(features, labels, False)
            metric.add(
                *[criterion(preds, labels) for criterion in criterion_a],
                *[ls * len(features) for ls in ls_es],
                len(features)
            )
            pbar.update(1)
        # 生成测试日志
        log = {}
        i = 0
        for i, computer in enumerate(criterion_a):
            log['valid_' + ptools.get_computer_name(computer)] = metric[i] / metric[-1]
        i += 1
        for j, loss_name in enumerate(l_names):
            log['valid_' + loss_name] = metric[i + j] / metric[-1]
        return log

    def __train_with_multithreading(self, train_iter, valid_iter=None) -> History:
        """多进程训练实现
        :param train_iter: 训练数据迭代器
        :param valid_iter: 验证数据迭代器
        :param pbar: 进度条
        :return: 训练历史记录
        """
        # 提取训练器参数
        pbar = self.pbar
        n_epochs = self.hps['epochs']
        history = None

        pbar.set_description('\r正在创建队列和事件对象……')
        # 使用进程池处理训练进程和记录进程
        pbar_Q = PQueue()
        data_Q = PQueue()
        eval_Q = PQueue()
        log_Q = PQueue()
        end_env = PEvent()
        # 将无法pickle的对象进行特殊序列化
        pbar.set_description('\r正在开启子进程……')
        # self.module = dill.dumps(self.module)
        del self.pbar
        # 生成子进程并开启
        process_pool = []
        try:
            data_subprocess = Process(
                target=data_iter_impl,
                args=(n_epochs, dill.dumps(train_iter), data_Q, end_env)
            )
            process_pool = [data_subprocess]
            data_subprocess.start()
            pbar.set_description('\r数据加载子进程已开启')
            train_subprocess = Process(
                target=train_impl,
                args=(dill.dumps(self), pbar_Q, eval_Q, log_Q, data_Q, end_env)
            )
            process_pool += [train_subprocess]
            train_subprocess.start()
            pbar.set_description('\r训练子进程已开启')
            # 如果self携带有网络，则将网络对象解绑以减少内存消耗
            module = self.module
            self.module = None
            eval_subprocess = Process(
                target=eval_impl,
                args=(module.loss_names, self.criterion_a, pbar_Q, eval_Q, log_Q, end_env)
            )
            process_pool += [eval_subprocess]
            # 日志进程
            if valid_iter:
                log_subprocess = Process(
                    target=vlog_impl,
                    args=(self, dill.dumps(valid_iter), pbar_Q, log_Q, end_env)
                )
            else:
                log_subprocess = Process(
                    target=tlog_impl,
                    args=(self, pbar_Q, log_Q, end_env)
                )
            process_pool += [log_subprocess]
            for p in process_pool:
                if not p.is_alive():
                    p.start()
            pbar.set_description('\r全部子进程已开启')
            self.module = module
            # 接受进度条队列消息
            while True:
                item = pbar_Q.get()
                if item is None:
                    break
                elif isinstance(item, Exception):
                    raise InterruptedError('训练过程中某处触发了异常，请根据上条Trackback信息进行排查！')
                elif isinstance(item, int):
                    pbar.update(item)
                elif isinstance(item, str):
                    pbar.set_description(item)
                elif isinstance(item, History):
                    history = item
                    break
                else:
                    raise ValueError(f'不识别的信号{item}')
            for p in process_pool:
                if p.is_alive():
                    p.join()
        except Exception as e:
            for p in process_pool:
                if p.is_alive():
                    p.terminate()
            raise e
        self.pbar = pbar
        if history is None:
            raise RuntimeError('历史记录对象丢失！')
        return history

    def train_with_profiler(self, data_iter, log_path):
        # 提取训练器参数
        k = self.hps['k']
        n_epochs = 2
        # 取相对较少数量的那个数据迭代器进行性能测试
        if k > 1:
            _, data_iter = next(data_iter)
        else:
            # 提取训练迭代器和验证迭代器
            data_iter = list(data_iter)
            if len(data_iter) == 2:
                _, data_iter = [it[0] for it in data_iter]
            elif len(data_iter) == 1:
                data_iter, _ = data_iter[0][0], None
            else:
                raise ValueError(f"无法识别的数据迭代器，其提供的长度为{len(data_iter)}")

        # 进度条设置
        self.pbar = tqdm(
            total=len(data_iter) * n_epochs, unit='批', position=0,
            desc=f'正在进行训练准备……', mininterval=1, ncols=100
        )
        profiling_impl(n_epochs, os, log_path, self, data_iter)