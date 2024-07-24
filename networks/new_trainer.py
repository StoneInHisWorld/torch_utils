import traceback
import warnings
from typing import Iterable

import dill
import torch
from torch import nn
from torch.multiprocessing import Event as PEvent
from torch.multiprocessing import Process
from torch.multiprocessing import SimpleQueue as PQueue
from torchsummary import summary
from tqdm import tqdm

from networks.new_basic_nn import BasicNN
from utils.accumulator import Accumulator
from utils.history import History


class net_builder:

    def __init__(self, destroy=False):
        """网络创建装饰器，创建好网络后传输给需要的函数。请使用@new_builder()进行调用

        需要修饰的方法的前三个参数指定为net_class, n_init_args, n_init_kwargs

        需要对summary_input_size、summary_batch_size、print_net进行修改以进行
        torchsummary的网络信息打印。三个变量含义分别是输入形状、输入批量数、是否进行summary，
        默认值分别为(256, 256)、16、False
        :param destroy: 是否在函数调用过后销毁网络，以节省内存
        """
        self.destroy = destroy

    def __call__(self, func):

        def wrapper(*args, **kwargs):
            _, trainer, net_class, n_init_args, n_init_kwargs, *args = args
            # 构造网络对象并交由trainer持有
            net = self.__build(net_class, n_init_args, n_init_kwargs, trainer)
            setattr(trainer, 'module', net)
            args = trainer, net, *args
            args = func(_, *args, **kwargs)

            return args

        return wrapper

    def __build(self, net_class, n_init_args, n_init_kwargs, trainer):
        """根据参数构造一个神经网络
        :param net_class: 构造的神经网络类型，作为类构造器传入
        :param n_init_args: 类构造器所用位置参数
        :param n_init_kwargs: 类构造器所用关键字参数
        :return: 构造完成的神经网络
        """
        assert issubclass(net_class, BasicNN), '请使用BasicNN的子类作为训练网络！'
        print(f'\r正在构造{net_class.__name__}', end='', flush=True)
        net = net_class(*n_init_args, **n_init_kwargs)
        print(f'\r构造{net_class.__name__}完成')
        self.__list_net(net, trainer)
        return net

    def __list_net(self, net, trainer) -> None:
        """打印网络信息。
        :param net: 待打印的网络
        :return: None
        """
        if trainer.runtime_cfg['print_net']:
            input_size = trainer.datasource.fea_channel, *net.required_shape
            try:
                summary(
                    net, input_size=input_size,
                    batch_size=trainer.hps['batch_size']
                )
            except RuntimeError as _:
                print(net)


class prepare:

    def __init__(self, what='train'):
        """函数准备装饰器，进行不同类型的函数准备，请使用@prepare()进行调用

        'train': 需要传入参数net、prepare_args、criterion_a、*args，其含义分别为训练待使用的网络对象，
        准备动作所需位置参数，训练所需评价指标函数，训练函数其他位置参数。训练后会将优化器、学习率规划器以及损失函数销毁

        'valid': 需要传入参数net、pbar、criterion_a、*args，其含义分别为验证待使用的网络对象，
        验证所用进度条，验证所需评价指标函数，验证函数其他位置参数

        'test': 需要传入参数net、criterion_a、ls_fn_args、*args，其含义分别为测试待使用的网络对象，
        测试所需评价指标函数，测试所用损失函数参数，测试函数其他位置参数

        :param what: 进行何种准备
        """
        self.what = what

    def __call__(self, func):

        def wrapper(*args, **kwargs):
            # trainer, *args = args
            if self.what == 'train':
                args = self.__prepare_train(*args)
                # 执行本体函数
                result = func(*args, **kwargs)
                self.__after_train(args[0])
            elif self.what == 'valid':
                args = self.__prepare_valid(*args)
                # 执行本体函数
                with torch.no_grad():
                     result = func(*args, **kwargs)
            elif self.what == 'test':
                args = self.__prepare_test(*args)
                # 执行本体函数
                with torch.no_grad():
                    result = func(*args, **kwargs)
                self.__after_test(args[0])
            else:
                raise ValueError(f'未知准备类型{self.what}!')
            return result

        return wrapper

    @net_builder()
    def __prepare_train(self, *args):
        trainer, net, prepare_args, *args = args
        # 训练前准备
        self.optimizer_s, self.scheduler_s, _ = net.prepare_training(*prepare_args)
        trainer.pbar.set_description('训练中……')
        args = trainer, self.optimizer_s, self.scheduler_s, *args
        return args

    def __after_train(self, trainer):
        net = trainer.module
        # 清除训练痕迹
        del net.optimizer_s, net.scheduler_s, net.ls_fn_s, \
            net.lr_names, net.loss_names, net.test_ls_names

    @staticmethod
    def __prepare_valid(*args):
        trainer, *args = args
        trainer.module.eval()
        trainer.pbar.set_description('验证中……')
        return trainer, *args

    def __prepare_test(self, *args):
        trainer, *args = args
        net = trainer.module
        net.prepare_training(ls_args=trainer.prepare_args[2])
        del net.optimizer_s, net.scheduler_s
        net.eval()
        return trainer, *args

    def __after_test(self, trainer):
        net = trainer.module
        # 清除测试痕迹
        del net.ls_fn_s, net.lr_names, net.loss_names, net.test_ls_names


def data_iter_subpro_impl(
        n_epochs, data_iter, data_Q: PQueue, end_env: PEvent
):
    """
    进行data_Q.put()
    :param n_epochs:
    :param data_iter:
    :param data_Q: 数据供给队列，需要给定长度以节省内存
    :return:
    """
    try:
        data_iter = dill.loads(data_iter)
        # 训练主循环
        for epoch in range(n_epochs):
            data_Q.put(f'{epoch + 1}/{n_epochs}')
            for batch in data_iter:
                data_Q.put(batch)
                if end_env.is_set():
                    raise InterruptedError('获取数据时被中断！')
            # 放置None作为世代结束标志
        # print('data_fetching ends')
        data_Q.put(None)
        print('data_iter_subpro_impl ends')
        end_env.wait()
    except Exception as e:
        traceback.print_exc()
        data_Q.put(e)
        print('data_iter_subpro_impl ends')


def train_subprocess_impl(
        net_init, net_init_args, net_init_kwargs,  # 网络设置
        optimizer_s, lr_scheduler_s,  # 训练设置
        pbar_Q: PQueue, log_Q: PQueue, data_Q: PQueue,  # 队列
        data_end_env: PEvent
):
    """
    进行data_Q.get()、pbar_Q.put()以及log_Q.put()
    :param net:
    :param optimizer_s:
    :param lr_scheduler_s:
    :param pbar_Q:
    :param log_Q:
    :param data_Q:
    :param data_end_env:
    :return:
    """
    try:
        # net = dill.loads(net)
        print('\r正在创建模型副本', end='', flush=True)
        net = net_init(*net_init_args, **net_init_kwargs)
        print('\r模型副本创建完成', end='', flush=True)
        net.train()
        while True:
            item = data_Q.get()
            # 收到了None，认为是数据供给结束标志
            if item is None:
                break
            # 收到了异常，则抛出
            elif isinstance(item, Exception):
                raise item
            # 收到了字符串，认为是世代结束标志
            elif isinstance(item, str):
                pbar_Q.put(f'世代{item} 训练中……')
                if item.startswith('1'):
                    # 如果是第一次世代更新，则只放入学习率组
                    log_Q.put([
                        [
                            [param['lr'] for param in optimizer.param_groups]
                            for optimizer in optimizer_s
                        ],
                        # ], True)
                    ])
                else:
                    # log队列中放入每个优化器的学习率组以及网络参数
                    log_Q.put([
                        [
                            [param['lr'] for param in optimizer.param_groups]
                            for optimizer in optimizer_s
                        ],
                        net.state_dict()
                        # ], True)
                    ])
                    # 更新学习率变化器组
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UserWarning)
                        for scheduler in lr_scheduler_s:
                            scheduler.step()
            # 收到了元组，则认为是数据批次
            elif isinstance(item, tuple):
                X, y = item
                net.train()
                # 得到预测值以及损失组，即对应每个损失函数的损失值
                pred, ls_es = net.forward_backward(X, y)
                ls_es = [ls.detach() for ls in ls_es]
                # 将网络参数、预测值、损失值、标签值作为信号放入队列中
                # log_Q.put((pred.detach(), ls_es, y.detach().clone()), True)
                log_Q.put((pred.detach(), ls_es, y.detach().clone()))
                # 完成了一批数据的计算，更新进度条
                pbar_Q.put(1)
            else:
                raise ValueError(f'不识别的信号{item}！')
        # 通知data_iter进程结束，通知log_process结束
        # data_end_env.set()
        # log_Q.put(None, True)
        log_Q.put(None)
        # 等待记录进程结束
        # log_end_env.wait()
        print('train_subpro_impl ends')
        data_end_env.wait()
        return
    except Exception as e:
        traceback.print_exc()
        log_Q.put(e)
        print('train_subpro_impl ends')


@torch.no_grad()
def tv_log_subprocess_impl(
        net_init, net_init_args, net_init_kwargs,  # 网络相关
        valid_iter,  # 数据相关
        criterion_a,  # 计算相关
        history, lr_names, cri_los_names,  # 记录对象
        log_Q: PQueue, pbar_Q: PQueue, end_env: PEvent  # 信号量相关
):
    """在多线程处理中，训练以及验证数据的记录实现。
    进行log_Q.get()
    :param net: 网络结构
    :param valid_iter: 验证数据迭代器
    :param criterion_a: 准确率计算函数
    :param history: 历史记录
    :param lr_names: 学习率名称，用于记录每个世代的学习率
    :param cri_los_names: 评价指标和损失值名称，用于记录每个世代的评价指标和损失值组合
    :param log_Q: 数据交换队列，train_impl通过该队列传输需要记录的信息
    :return: history历史记录
    """
    try:
        print('\r正在创建模型副本', end='', flush=True)
        net = net_init(*net_init_args, **net_init_kwargs)
        print('\r模型副本创建完成', end='', flush=True)
        net.eval()
        # 创建训练模型的副本
        valid_iter = dill.loads(valid_iter)
        metric = Accumulator(len(cri_los_names) + 1)
        while True:
            # 从队列中获取信号
            # item = log_Q.get(True)
            item = log_Q.get()
            # 收到了None，则认为是训练完成信号
            if item is None:
                break
            # 将异常抛出
            elif isinstance(item, Exception):
                raise item
            # 如果收到了序列，则认为是世代结束标志
            elif isinstance(item, list):
                if len(item) == 2:
                    # 若是获取到了学习率组和网络参数，则认为完成了一个世代的迭代，刷新累加器并进行验证
                    lr_group, state_dict = item
                    # 进行验证
                    net.load_state_dict(state_dict)
                    valid_log = valid_subprocess_impl(net, valid_iter, criterion_a, pbar_Q)
                    # 记录训练和验证数据
                    history.add(
                        cri_los_names + list(valid_log.keys()),
                        [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
                        list(valid_log.values())
                    )
                    metric.reset()
                    del state_dict
                else:
                    [lr_group] = item
                # 记录学习率
                history.add(lr_names, lr_group)
                del lr_group
            # 如果收到了元组，则认为是迭代结束标志
            elif isinstance(item, tuple):
                # 分解成预测值、损失值、标签值，进行评价指标的计算以及损失值的累加
                pred, ls_es, y = item
                # 计算训练准确率数据
                correct_s = []
                for criterion in criterion_a:
                    correct = criterion(pred, y)
                    correct_s.append(correct)
                num_examples = y.shape[0]
                metric.add(
                    *correct_s, *[ls * num_examples for ls in ls_es],
                    num_examples
                )
                del pred, ls_es, y
            else:
                raise NotImplementedError(f'无法识别的信号{item}！')
            del item
        # 将结果放在进度条队列中
        pbar_Q.put(history)
        print('tv_log_subprocess_impl ends')
        end_env.set()
        return
    except Exception as e:
        traceback.print_exc()
        pbar_Q.put(e)
        print('tv_log_subprocess_impl ends')


@torch.no_grad()
def valid_subprocess_impl(
        net, valid_iter, criterion_a, pbar_Q
):
    """
    进行pbar_Q.put()
    :param net:
    :param valid_iter:
    :param criterion_a:
    :param pbar_Q:
    :return:
    """
    net.eval()
    pbar_Q.put('验证中……')
    # 要统计的数据种类数目
    criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
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
        pbar_Q.put(1)
    # 生成测试日志
    log = {}
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


class Trainer:

    def __init__(self,
                 module_class, m_init_args, m_init_kwargs, prepare_args,  # 网络模型相关
                 datasource, criterion_a, hps, runtime_cfg,  # 训练、验证、测试依赖参数
                 with_hook=False, hook_mute=True):
        """本训练器包含有网络初始化的功能。
        :param module:
        :param optimizer_s:
        :param lr_scheduler_s:
        :param acc_fn:
        :param n_epochs:
        :param ls_fn:
        :param with_hook:
        :param hook_mute:
        """
        assert issubclass(module_class, BasicNN), f'模型类型{module_class}未知，请使用net包下实现的网络模型'
        # 存放网络初始化参数
        self.module_class = module_class
        self.__m_init_args = m_init_args
        self.__m_init_kwargs = m_init_kwargs
        self.prepare_args = prepare_args
        self.module = None
        # 设置训练、验证、测试依赖参数
        self.datasource = datasource
        self.hps = hps
        self.runtime_cfg = runtime_cfg
        self.criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
        self.pbar = None
        # 处理hook机制
        self.with_hook = with_hook
        self.hook_mute = hook_mute
        self.train = self.hook(self.train)

    def train(self, data_iter) -> History:
        """训练公共接口，根据训练器自身参数，以及接口接收参数自动判断进行的训练类型
        :param prepare_args: 训练准备参数，请指定为(优化器训练参数，学习率规划器参数，损失函数创建参数)
        :param criterion_a: 训练所用评价指标函数
        :param data_iter: 训练所用数据迭代器
        :param hps: 训练所需超参数
        :param runtime_cfg: 训练所需运行配置参数
        :return:
        """
        n_workers = self.runtime_cfg['n_workers']
        n_epochs = self.hps['epochs']
        # 判断是否是k折训练
        if self.hps['k'] > 1:
            raise NotImplementedError('k折训练尚未实现！')
            # 是否进行k-折训练
            # history = self.__train_with_k_fold(data_iter, k, n_workers)
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
            if n_workers <= 3:
                # 不启用多线程训练
                if valid_iter is not None:
                    # 进行训练和验证
                    train_fn, train_args = self.__train_and_valid, (
                        self.module_class, self.__m_init_args, self.__m_init_kwargs,
                        self.prepare_args, train_iter, valid_iter
                    )
                else:
                    raise NotImplementedError('简单训练尚未实现！')
                    # history = self.__train(train_iter)
            else:
                # 启用多进程训练
                raise NotImplementedError('多进程训练尚未实现！')
                # history = self.__train_with_subprocesses(train_iter, valid_iter, None)
        # 设置进度条
        self.pbar = tqdm(
            total=pbar_len, unit='批', position=0,
            desc=f'正在进行训练准备……', mininterval=1
        )
        history = train_fn(*train_args)
        self.pbar.close()
        return history

    # def test(self, test_iter) -> dict:
    #     """测试公共接口，调用测试函数进行测试
    #     取对象构造时传入的prepare_args的2号位作为测试所用损失函数构造位置参数
    #     :param test_iter: 测试数据迭代器
    #     :return: 测试记录对象
    #     """
    #     return self.__test(
    #         self.module, self.criterion_a, self.prepare_args[2], test_iter
    #     )

    @prepare('test')
    def test(self, test_iter) -> dict:
        """测试实现，取出迭代器中的下一batch数据，预测后计算准确度和损失

        本函数实际签名为：
        def __test(
            net: BasicNN, criterion_a: List[Callable], ls_fn_args: tuple,
            prefix: str, test_iter: DataLoader
        ) -> dict

        其中ls_fn_args用于网络损失函数创建。损失函数会在函数运行完后进行销毁。

        :param net: 测试所用网络模型对象
        :param criterion_a: 计算准确度所使用的函数，该函数需要求出整个batch的准确率之和。
            签名需为：acc_func(Y_HAT, Y) -> float or torch.Tensor
        :param prefix: 测试记录项前缀
        :param test_iter: 测试数据迭代器
        :return: 测试记录
        """
        # 提取训练器参数
        net = self.module
        criterion_a = self.criterion_a
        pbar = tqdm(test_iter, unit='批', position=0, desc=f'测试中……', mininterval=1)
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
            try:
                log['test_' + computer.__name__] = metric[i] / metric[-1]
            except AttributeError:
                log['test_' + computer.__class__.__name__] = metric[i] / metric[-1]
        i += 1
        for j, loss_name in enumerate(l_names):
            log['test_' + loss_name] = metric[i + j] / metric[-1]
        return log

    # @staticmethod
    @prepare('valid')
    def __valid(self, valid_iter) -> [float, float]:
        """验证函数实现，取出迭代器中的下一batch数据，预测后计算准确度和损失

        本函数的真正签名为：
        def __valid(
            net, pbar, criterion_a, valid_iter
        ) -> dict

        @prepare装饰器会生成prefix前缀传递给本函数
        :param net: 测试所用网络对象
        :param pbar: 测试所用进度条
        :param criterion_a: 测试
        :param valid_iter: 测试所用数据迭代器
        :return: 测试记录字典
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
            try:
                log['valid_' + computer.__name__] = metric[i] / metric[-1]
            except AttributeError:
                log['valid_' + computer.__class__.__name__] = metric[i] / metric[-1]
        i += 1
        for j, loss_name in enumerate(l_names):
            log['valid_' + loss_name] = metric[i + j] / metric[-1]
        return log

    def __train_with_subprocesses(self,
                                  train_iter, valid_iter=None,
                                  pbar=None) -> History:
        """多进程训练实现
        :param train_iter: 训练数据迭代器
        :param valid_iter: 验证数据迭代器
        :param pbar: 进度条
        :return: 训练历史记录
        """
        # 提取基本训练对象
        net = self.module
        n_epochs = self.__n_epochs
        criterion_a = self.__criterion_a if isinstance(self.__criterion_a, list) else [self.__criterion_a]
        progress_len = (len(train_iter) + len(valid_iter)) * n_epochs if valid_iter else len(train_iter) * n_epochs
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
        # 学习率项
        lr_names = net.lr_names
        optimizer_s = self.__optimizer_s
        lr_scheduler_s = self.__lr_scheduler_s
        criterion_a = self.__criterion_a
        # 设置历史记录对象
        history = History(*(criteria_names + loss_names + lr_names))

        print('\r正在创建队列和事件对象……', end='', flush=True)
        # 使用进程池处理训练进程和记录进程
        pbar_Q = PQueue()
        log_Q = PQueue()
        data_Q = PQueue()
        data_end_env = PEvent()
        # log_end_env = PEvent()
        # 将无法pickle的对象进行特殊序列化
        print('\r正在进行特殊序列化……', end='', flush=True)
        train_iter = dill.dumps(train_iter)
        valid_iter = dill.dumps(valid_iter) if valid_iter else None
        # net_copy = dill.dumps(deepcopy(net))
        # net_copy = deepcopy(net)
        net_init_args, net_init_kwargs = net.get_clone_function()
        net_init = net.__class__
        net = dill.dumps(net)
        # del net
        print('\r子进程创建准备完毕')

        # 设置进度条
        if pbar is None:
            # 如果调用函数不提供进度条，则创建一个新的进度条
            pbar = tqdm(total=progress_len, unit='批', desc='训练中……', mininterval=1)
        # 生成子进程
        data_subprocess = Process(
            target=data_iter_subpro_impl,
            args=(n_epochs, train_iter, data_Q, data_end_env)
        )
        train_subprocess = Process(
            target=train_subprocess_impl,
            args=(
                net_init, net_init_args, net_init_kwargs,
                optimizer_s, lr_scheduler_s,
                pbar_Q, log_Q, data_Q, data_end_env
            )
        )
        if valid_iter:
            log_subprocess = Process(
                target=tv_log_subprocess_impl,
                args=(
                    net_init, net_init_args, net_init_kwargs,
                    valid_iter, criterion_a,
                    history, lr_names, criteria_names + loss_names,
                    log_Q, pbar_Q, data_end_env
                )
            )
        else:
            raise NotImplementedError('暂未编写单训练过程')
        process_pool = [data_subprocess, train_subprocess, log_subprocess]
        # 实时监控各项任务的执行情况
        try:
            for p in process_pool:
                p.start()
            while True:
                # item = pbar_Q.get(True)
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
                p.join()
        except Exception as e:
            for p in process_pool:
                if p:
                    p.terminate()
            raise e
        pbar.close()
        return history

    # @staticmethod
    # @net_builder()
    @prepare('train')
    def __train_and_valid(self,
                          optimizer_s, scheduler_s,
                          train_iter, valid_iter
                          ) -> History:
        """神经网络训练和验证实现函数。

        本函数的实际签名为：

        def __train_and_valid(
            module_class: type, __m_init_args: tuple, __m_init_kwargs: dict,
            prepare_args: tuple, criterion_a: list[Callable],
            train_iter: DataLoader, valid_iter: DataLoader,
            n_epochs: int
        ) -> BasicNN, History

        module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
        prepare_args用于给prepare装饰器进行网络训练准备

        :param optimizer_s: 优化器组合
        :param scheduler_s: 学习率规划期组合
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
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
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

    def __train(self, train_iter) -> History:
        """神经网络训练函数。
        :return: 训练数据记录`History`对象
        """
        net = self.module
        n_epochs = self.__n_epochs
        criterion_a = self.__criterion_a if isinstance(self.__criterion_a, list) else [self.__criterion_a]
        non_blocking = self.device.type == 'cuda' and train_iter.pin_memory
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [f'train_{criterion.__name__}' for criterion in criterion_a]
        # 学习率项
        lr_names = net.lr_names
        history = History(*(criteria_names + loss_names + lr_names))
        with tqdm(total=len(train_iter) * n_epochs, unit='批', position=0,
                  desc=f'训练中……', mininterval=1) as pbar:
            for epoch in range(n_epochs):
                pbar.set_description(f'世代{epoch + 1}/{n_epochs} 训练中……')
                history.add(
                    lr_names, [
                        [param['lr'] for param in optimizer.param_groups]
                        for optimizer in self.__optimizer_s
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
                for scheduler in self.__lr_scheduler_s:
                    scheduler.step()
                # TODO:记录训练数据，可以细化到每个批次
                history.add(
                    criteria_names + loss_names,
                    [metric[i] / metric[-1] for i in range(len(metric) - 1)]
                )
            pbar.close()
        return history

    def __train_with_k_fold(self, train_loaders_iter,
                            k: int = 10, n_workers=1) -> History:
        """
        使用k折验证法进行模型训练
        :param n_workers: 使用的处理机数量
        :param train_loaders_iter: 数据加载器供给，提供k折验证的每一次训练所需训练集加载器、验证集加载器
        :param k: 将数据拆分成k折，每一折轮流作验证集，余下k-1折作训练集
        :return: k折训练记录，包括每一折训练时的('train_l', 'train_acc', 'valid_l', 'valid_acc')
        """
        k_fold_history = None
        with tqdm(total=k * self.__n_epochs, position=0, leave=True, unit='批', mininterval=1) as pbar:
            for i, (train_iter, valid_iter) in enumerate(train_loaders_iter):
                pbar.set_description(f'\r训练折{pbar.n}……')
                # 计算训练批次数
                pbar.total = k * self.__n_epochs * (len(train_iter) + len(valid_iter))
                if n_workers <= 1:
                    history = self.__train_and_valid(train_iter, valid_iter, pbar)
                else:
                    history = self.__train_with_subprocesses(train_iter, valid_iter, pbar)
                if k_fold_history is None:
                    k_fold_history = history
                else:
                    k_fold_history += history
                pbar.set_description(f'\r折{i + 1}训练完毕')
        return k_fold_history

    def __deal_with_hook(self, net: Iterable[nn.Module]):
        self.__last_forward_output, self.__last_backward_data = {}, {}
        forward_handlers = []
        backward_handlers = []

        def __hook_forward_fn(module, input, output):
            if self.hook_mute:
                try:
                    self.__last_forward_output.pop(module)
                except Exception as _:
                    pass
            else:
                print(f'{module.__class__.__name__}前传')
                try:
                    last_input, last_output = self.__last_forward_output.pop(module)
                except Exception as _:
                    pass
                else:
                    flag = True
                    for li, i in zip(last_input, input):
                        flag = torch.equal(li, i) and flag
                    print(f'输入相等: {flag}')
                    flag = True
                    for lo, o in zip(last_output, output):
                        flag = torch.equal(lo, o) and flag
                    print(f'输出相等: {flag}')
                    print('-' * 20)
            # 记录模块的梯度
            self.__last_forward_output[module] = input, output
            return output

        def __hook_backward_fn(module, grad_input, grad_output):
            if self.hook_mute:
                try:
                    last_input, last_output = self.__last_backward_data.pop(module)
                except Exception as _:
                    pass
                else:
                    for li, i in zip(last_input, grad_input):
                        if li is None or i is None:
                            print(f'{module.__class__.__name__}反向传播中，{li}或{i}出现了None梯度')
                    for lo, o in zip(last_output, grad_output):
                        if lo is None or o is None:
                            print(f'{lo}或{o}出现了None梯度')
            else:
                print(f'{module.__class__.__name__}反向传播')
                try:
                    last_input, last_output = self.__last_backward_data.pop(module)
                except Exception as _:
                    pass
                else:
                    flag = True
                    for li, i in zip(last_input, grad_input):
                        if li is None or i is None:
                            print(f'{module.__class__.__name__}反向传播中，{li}或{i}出现了None梯度')
                        else:
                            flag = torch.equal(li, i) and flag
                            print(f'输入梯度相等: {flag}')
                    flag = True
                    for lo, o in zip(last_output, grad_output):
                        if lo is None or o is None:
                            print(f'{lo}或{o}中出现了None梯度')
                        else:
                            flag = torch.equal(lo, o) and flag
                            print(f'输出梯度相等：{flag}')
                    print('-' * 20)
            self.__last_backward_data[module] = grad_input, grad_output

        for m in net:
            forward_handlers.append(m.register_forward_hook(hook=__hook_forward_fn))
            backward_handlers.append(m.register_full_backward_hook(hook=__hook_backward_fn))
        return forward_handlers, backward_handlers

    def hook(self, func):

        def wrapper(*args, **kwargs):
            if self.with_hook:
                self.__f_handles, self.__b_handles = self.__deal_with_hook(self.module)
            ret = func(*args, **kwargs)
            if self.with_hook:
                for handle in self.__f_handles + self.__b_handles:
                    handle.remove()
            return ret

        return wrapper
