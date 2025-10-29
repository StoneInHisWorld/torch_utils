import os
import torch
from tqdm import tqdm

from networks import BasicNN
from networks.decorators import prepare
from networks.trainer.__hook_impl import hook
from networks.trainer.__profiler_impl import profiling_impl
from utils.accumulator import Accumulator
from utils.func import pytools as ptools
from utils.history import History
from threading import Thread


def add_lr_to_history(net, history):
    lr_names, lrs = net.get_lr_groups()
    history.add([f"{ln}_lrs" for ln in lr_names], lrs)


def is_multiprocessing(n_workers):
    return n_workers >= 3


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
        if len(self.criterion_a) == 0:
            raise ValueError(f"训练器没有拿到训练指标方法!"
                             f"请检查自定义数据集类SelfDefinedDataSet中对于{module_class.__name__}的评价指标赋值。")
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
            if not is_multiprocessing(n_workers):
                # 不启用多线程训练
                if valid_iter is not None:
                    # 进行训练和验证
                    train_fn, train_args = self.__train_and_valid, (train_iter, valid_iter)
                else:
                    # 进行训练
                    train_fn, train_args = self.__train, (train_iter,)
            else:
                # 启用多进程训练
                # train_fn, train_args = self.__train_and_valid_with_preprocessing, (train_iter, valid_iter)
                # train_fn, train_args = self.__new_train_and_valid_with_preprocessing, (train_iter, valid_iter)
                train_fn, train_args = self.__pipe_train_and_valid_with_preprocessing, (train_iter, valid_iter)
        # 设置进度条
        pbar = tqdm(
            total=pbar_len, unit='批', position=0, desc=f'正在进行训练准备……', 
            mininterval=1, ncols=100, bar_format=""
        )
        self.pbar = pbar  # 多进程运行需要删除此属性，此举防止pbar被回收
        history = train_fn(*train_args)
        pbar.close()
        return history

    @prepare('predict')
    def predict(self, ret_ls_metric=True, ret_ds=True):
        """根据参数创建网络并进行预测
        对于数据迭代器中的每一批次数据，根据需要保存输入数据、预测数据、标签集、评价指标、损失值数据，并打包返回。

        Args:
            ret_ls_metric: 是否返回损失值数据以及评价指标数据
            ret_ds: 是否返回原数据集

        Returns:
            [预测数据]
             + [特征数据集，标签数据集] if ret_ls_metric == True
             + [评价指标数据，损失值数据，评价指标函数名称，
                损失值函数名称] if ret_ds == True
        """
        # 提取训练器参数
        net = self.module
        criterion_a = self.criterion_a
        pbar = self.pbar
        # 本次预测所产生的全部数据池
        inputs, predictions, labels, metrics, loss_pool = [], [], [], [], []
        # 对每个批次进行预测，并进行评价指标和损失值的计算
        for fe_batch, lb_batch in pbar:
            result = net.forward_backward(fe_batch, lb_batch, False)
            pre_batch, ls_es = result
            predictions.append(pre_batch)
            if ret_ds:
                inputs.append(fe_batch)
                labels.append(lb_batch)
            if ret_ls_metric:
                metrics.append(torch.vstack([
                    criterion(pre_batch, lb_batch, size_averaged=False)
                    for criterion in criterion_a
                ]).T)
                loss_pool.append(torch.vstack(ls_es).T)
        pbar.set_description('结果计算完成')
        # # 将所有批次的数据堆叠在一起
        # ret = [torch.cat(predictions, dim=0)]
        # if ret_ds:
        #     ret.append(torch.cat(inputs, dim=0))
        #     ret.append(torch.cat(labels, dim=0))
        # if ret_ls_metric:
        #     ret.append(torch.cat(metrics, dim=0))
        #     ret.append(torch.cat(loss_pool, dim=0))
        #     ret.append([
        #         ptools.get_computer_name(criterion) for criterion in criterion_a
        #     ])
        #     ret.append(net.test_ls_names)
        # return ret
        ret = [predictions]
        if ret_ds:
            ret += [inputs, labels]
        if ret_ls_metric:
            ret += [metrics, loss_pool, [
                ptools.get_computer_name(criterion) for criterion in criterion_a
            ], net.test_ls_names]
        return ret

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
        loss_names = [f'train_{item}' for item in net.train_ls_names]
        # 评价项
        criteria_names = [f'train_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
        # 学习率项
        lr_names = [f'{lr}_lrs' for lr in net.lr_names]
        history = History(*(criteria_names + loss_names + lr_names))
        for epoch in range(n_epochs):
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
            add_lr_to_history(net, history)
            # 记录批次训练损失总和，评价指标，样本数
            metric = Accumulator(len(loss_names + criteria_names) + 1)
            # 训练主循环
            for X, y in train_iter:
                net.train()
                preds, ls_es = net.forward_backward(X, y)
                with torch.no_grad():
                    num_examples = len(preds)
                    correct_s = []
                    for criterion in criterion_a:
                        correct = criterion(preds, y)
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
        # 损失项
        trls_names = [f'train_{item}' for item in net.train_ls_names]
        # 评价项
        criteria_names = [f'train_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
        # 学习率项
        lr_names = [f'{lr}_lrs' for lr in net.lr_names]
        history = History(*(criteria_names + trls_names + lr_names))
        # 世代迭代主循环
        for epoch in range(n_epochs):
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
            # history.add(*net.get_lr_groups())
            add_lr_to_history(net, history)
            # 记录批次训练损失总和，评价指标，样本数
            metric = Accumulator(len(trls_names + criteria_names) + 1)
            # 训练主循环
            for X, y in train_iter:
                pred, ls_es = net.forward_backward(X, y)
                with torch.no_grad():
                    num_examples = len(pred)
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
            net.update_lr()
            valid_log = self.__valid(valid_iter)
            # 记录训练数据
            history.add(
                criteria_names + trls_names + list(valid_log.keys()),
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
            num_samples = len(preds)
            metric.add(
                *[criterion(preds, labels) for criterion in criterion_a],
                *[ls * num_samples for ls in ls_es], len(preds)
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
        net.eval()
        for features, labels in valid_iter:
            preds, ls_es = net.forward_backward(features, labels, False)
            num_samples = len(preds)
            metric.add(
                *[criterion(preds, labels) for criterion in criterion_a],
                *[ls * num_samples for ls in ls_es], num_samples
            )
            pbar.update(1)
        # 生成测试日志
        log = {}
        i = 0
        for i, computer in enumerate(criterion_a):
            log['valid_' + ptools.get_computer_name(computer)] = metric[i] / metric[-1]
        i += 1
        for j, ln in enumerate(l_names):
            log['valid_' + ln] = metric[i + j] / metric[-1]
        net.train()
        return log

    def __train_and_valid_with_preprocessing(self, train_iter, valid_iter) -> History:
        """多进程训练实现
        :param train_iter: 训练数据迭代器
        :param valid_iter: 验证数据迭代器
        :return: 训练历史记录
        """
        from __subprocess_impl import train_valid_impl
        # 提取训练器参数
        pbar = self.pbar
        del self.pbar
        n_epochs = self.hps['epochs']

        pbar.set_description('\r正在创建队列和事件对象……')
        # 进程通信队列
        # TODO：改成双工Pipe实现
        ctx = torch.multiprocessing.get_context("spawn")
        tdata_q = ctx.Queue(int(self.runtime_cfg['train_prefetch']))  # 传递训练数据队列
        vdata_q = ctx.Queue(int(self.runtime_cfg['valid_prefetch']))  # 传递验证数据队列
        pbar_q = ctx.Queue()  # 传递进度条更新消息队列
        epoch_q = ctx.Queue()  # 传递世代更新消息队列

        def update_pbar():
            msg = pbar_q.get()
            while msg:
                assert isinstance(msg, int) or isinstance(msg, str), "进度条更新只接受数字或字符串更新！"
                if isinstance(msg, int):
                    pbar.update(msg)
                else:
                    pbar.set_description(msg)
                msg = pbar_q.get()

        # 生成子进程用于创建网络、执行网络更新并记录数据
        # 搭建输出结果通信管道
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        # 创建子线程进行训练和验证操作，并更新进度条
        tv_subp = ctx.Process(target=train_valid_impl, args=(
            self, tdata_q, vdata_q, pbar_q, epoch_q, child_conn
        ))
        pbar_update_thread = Thread(target=update_pbar)  # 更新进度条
        # 开启两个子进程
        pbar_update_thread.start()
        tv_subp.start()
        # 获取所有的数据，并且发送给训练进程
        for epoch in range(n_epochs):
            # 通知子进程新的世代开始了
            epoch_q.put(epoch)
            pbar.set_description(f'获取世代{epoch + 1}/{n_epochs}的训练数据……')
            # 不断从训练数据集迭代器中取训练数据
            for X, y in train_iter:
                tdata_q.put((X, y))
            # 通知训练进程，当前世代的数据已经传递完毕
            tdata_q.put(None)
            pbar.set_description(f'获取世代{epoch + 1}/{n_epochs}的验证数据……')
            # 从验证迭代器中取验证数据
            for X, y in valid_iter:
                vdata_q.put((X, y))
            # 通知验证进程，当前世代的数据已经传递完毕
            vdata_q.put(None)
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 数据获取完毕，等待网络消耗剩下的数据')
        # 使用None通知子进程数据已经获取完毕
        # tdata_q.put(None)
        # vdata_q.put(None)
        epoch_q.put(None)
        # 处理随机顺序返回的结果
        ret = [parent_conn.recv(), parent_conn.recv()]
        if isinstance(ret[0], History) and isinstance(ret[1], BasicNN):
            history, self.module = ret
        elif isinstance(ret[0], BasicNN) and isinstance(ret[1], History):
            self.module, history = ret
        else:
            raise ValueError(f"多进程管道接收到了异常的数据类型，为{type(ret[0])}和{type(ret[1])}")
        return history

    def __new_train_and_valid_with_preprocessing(self, train_iter, valid_iter) -> History:
        """多进程训练实现
        :param train_iter: 训练数据迭代器
        :param valid_iter: 验证数据迭代器
        :return: 训练历史记录
        """
        from networks.trainer.__subprocess_impl import train_valid_impl
        
        # 提取训练器参数
        pbar = self.pbar
        del self.pbar
        n_epochs = self.hps['epochs']

        pbar.set_description('\r正在创建队列和事件对象……')
        # 进程通信队列
        # TODO：改成双工Pipe实现
        ctx = torch.multiprocessing.get_context("spawn")
        tdata_q = ctx.Queue(int(self.runtime_cfg['train_prefetch']))  # 传递训练数据队列
        vdata_q = ctx.Queue(int(self.runtime_cfg['valid_prefetch']))  # 传递验证数据队列
        pbar_q = ctx.Queue()  # 传递进度条更新消息队列
        epoch_q = ctx.Queue()  # 传递世代更新消息队列

        def update_pbar():
            msg = pbar_q.get()
            while msg is not None:
                assert isinstance(msg, int) or isinstance(msg, str), "进度条更新只接受数字或字符串更新！"
                if isinstance(msg, int):
                    pbar.update(msg)
                else:
                    pbar.set_description(msg)
                msg = pbar_q.get()
                # print(msg)

        def send_data(data_iter, data_q, epoch, which):
            pbar.set_description(f'获取世代{epoch}/{n_epochs}的{which}数据……')
            for X, y in data_iter:
                data_q.put((X, y))
            data_q.put(None)

        # 生成子进程用于创建网络、执行网络更新并记录数据
        # 搭建输出结果通信管道
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        # 创建子线程进行训练和验证操作，并更新进度条
        tv_subp = ctx.Process(target=train_valid_impl, args=(
            self, tdata_q, vdata_q, pbar_q, epoch_q, child_conn
        ))
        pbar_update_thread = Thread(target=update_pbar)  # 更新进度条
        # 开启两个子进程
        pbar_update_thread.start()
        tv_subp.start()
        # 获取所有的数据，并且发送给训练进程
        for epoch in range(1, n_epochs + 1):
            # 通知子进程新的世代开始了
            epoch_q.put(epoch)
            # 不断从迭代器中取数据
            tsending = Thread(target=send_data, args=(train_iter, tdata_q, epoch, "训练"))
            tsending.start()
            vsending = Thread(target=send_data, args=(valid_iter, vdata_q, epoch, "验证"))
            vsending.start()
            # 等待数据发送完毕
            tsending.join()
            vsending.join()
            pbar.set_description(f'世代{epoch}/{n_epochs} 数据获取完毕，等待网络消耗剩下的数据')
        # 使用None通知子进程数据已经获取完毕
        epoch_q.put(None)
        # 处理随机顺序返回的结果
        ret = [parent_conn.recv(), parent_conn.recv()]
        if isinstance(ret[0], History) and isinstance(ret[1], BasicNN):
            history, self.module = ret
        elif isinstance(ret[0], BasicNN) and isinstance(ret[1], History):
            self.module, history = ret
        else:
            raise ValueError(f"多进程管道接收到了异常的数据类型，为{type(ret[0])}和{type(ret[1])}")
        pbar_update_thread.join()
        tv_subp.join()
        return history

    def __pipe_train_and_valid_with_preprocessing(self, train_iter, valid_iter) -> History:
        """多进程训练实现
        :param train_iter: 训练数据迭代器
        :param valid_iter: 验证数据迭代器
        :return: 训练历史记录
        """
        from networks.trainer.__pipe_subprocess_impl import train_valid_impl 
        
        # 提取训练器参数
        pbar = self.pbar
        del self.pbar
        n_epochs = self.hps['epochs']

        pbar.bar_format = None  # 使用默认的进度条格式
        pbar.set_description('\r正在创建队列和事件对象……')
        # 进程通信队列
        ctx = torch.multiprocessing.get_context("spawn")
        tdata_pc, tdata_cc = ctx.Pipe(False)  # 传递训练数据队列
        vdata_pc, vdata_cc = ctx.Pipe(False)  # 传递验证数据队列
        pbar_q = ctx.Queue()  # 传递进度条更新消息队列，会用于记录进程间通信
        epoch_pc, epoch_cc = ctx.Pipe(duplex=False)  # 传递世代更新消息队列
        parent_conn, child_conn = ctx.Pipe(duplex=False)  # 搭建输出结果通信管道

        def update_pbar():
            msg = pbar_q.get()
            while msg is not None:
                assert isinstance(msg, int) or isinstance(msg, str), "进度条更新只接受数字或字符串更新！"
                if isinstance(msg, int):
                    pbar.update(msg)
                else:
                    pbar.set_description(msg)
                msg = pbar_q.get()

        def send_data(data_iter, data_q, epoch, which):
            pbar.set_description(f'获取世代{epoch + 1}/{n_epochs}的{which}数据……')
            for X, y in data_iter:
                data_q.send((X, y))
            data_q.send(None)

        # 生成子进程用于创建网络、执行网络更新并记录数据
        # 创建子线程进行训练和验证操作，并更新进度条
        tv_subp = ctx.Process(target=train_valid_impl, args=(
            self, tdata_pc, vdata_pc, pbar_q, epoch_pc, child_conn
        ))
        pbar_update_thread = Thread(target=update_pbar)  # 更新进度条
        # 开启两个子进程
        pbar_update_thread.start()
        tv_subp.start()
        # 获取所有的数据，并且发送给训练进程
        for epoch in range(1, n_epochs + 1):
            # 通知子进程新的世代开始了
            epoch_cc.send(epoch)
            # 不断从迭代器中取数据
            tsending = Thread(target=send_data, args=(train_iter, tdata_cc, epoch, "训练"))
            tsending.start()
            vsending = Thread(target=send_data, args=(valid_iter, vdata_cc, epoch, "验证"))
            vsending.start()
            # 等待数据发送完毕
            tsending.join()
            vsending.join()
            pbar.set_description(f'世代{epoch + 1}/{n_epochs} 数据获取完毕，等待网络消耗剩下的数据')
        # 使用None通知子进程数据已经获取完毕
        epoch_cc.send(None)
        # 处理随机顺序返回的结果
        ret = [parent_conn.recv(), parent_conn.recv()]
        if isinstance(ret[0], History) and isinstance(ret[1], BasicNN):
            history, self.module = ret
        elif isinstance(ret[0], BasicNN) and isinstance(ret[1], History):
            self.module, history = ret
        else:
            raise ValueError(f"多进程管道接收到了异常的数据类型，为{type(ret[0])}和{type(ret[1])}")
        pbar_update_thread.join()
        tv_subp.join()
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
        profiling_impl(n_epochs, log_path, self, data_iter)
