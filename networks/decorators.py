import inspect
import warnings
from functools import wraps
from typing import Iterable

import dill
import torch
from torchinfo import summary
from tqdm import tqdm

from networks import BasicNN, configure_network


class net_builder:
    """网络创建装饰器。
    使用@new_builder()进行调用，创建好网络后传输给需要的函数。
    """

    def __init__(self, mute=False):
        """网络创建装饰器。
        使用@new_builder()进行调用，创建好网络后传输给需要的函数。
        需要修饰的方法的前四个参数指定为trainer, net_class, n_init_args, n_init_kwargs，
            其中trainer提供数据源信息，后三个参数提供网络构造参数。
        若不需要构造网络，则令trainer.module对象不为None即可，此时只需向args提供trainer参数。

        :param mute: 是否在函数调用过后销毁网络，以节省内存。Deprecated!
        """
        self.mute = mute

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """修饰器本体
            传递给被修饰的函数的参数为trainer, net, *args
            如果trainer.module为None，则调用__build函数构造神经网络，并交由trainer保存。

            :param args:
            :param kwargs:
            :return:
            """
            prepare_obj, trainer, *args = args
            if isinstance(trainer, bytes):
                # 如果trainer被序列化了
                trainer = dill.loads(trainer)
            if trainer.module is None:
                # 构造网络对象并交由trainer持有
                net_class, n_init_args, n_init_kwargs = trainer.module_class, trainer.m_init_args, trainer.m_init_kwargs
                net = self.__build(net_class, n_init_args, n_init_kwargs, trainer, prepare_obj.mute)
                setattr(trainer, 'module', net)
            else:
                net = trainer.module
            args = trainer, net, *args
            args = func(prepare_obj, *args, **kwargs)

            return args

        return wrapper

    def __build(self, net_class, n_init_args, n_init_kwargs, trainer, mute):
        """根据参数构造一个神经网络
        :param net_class: 构造的神经网络类型，作为类构造器传入
        :param n_init_args: 类构造器所用位置参数
        :param n_init_kwargs: 类构造器所用关键字参数
        :return: 构造完成的神经网络
        """
        assert issubclass(net_class, BasicNN), '请使用BasicNN的子类作为训练网络！'
        if not mute:
            print(f'\r正在构造{net_class.__name__}', end='', flush=True)
        try:
            n_init_kwargs['with_checkpoint'] = trainer.runtime_cfg['with_checkpoint']
            net = net_class(*n_init_args, **n_init_kwargs)
        except FileNotFoundError:
            # 处理预训练网络加载
            # 去掉.ptsd
            where = n_init_kwargs['init_kwargs']['where'][:-5]
            try:
                net = torch.load(where + '.ptm')
            except FileNotFoundError:
                raise FileNotFoundError(f'找不到网络持久化文件{where + ".ptm"}')
        if not mute:
            print(f'\r构造{net_class.__name__}完成')
            self.__list_net(net, trainer)
        return net

    def __list_net(self, net, trainer) -> None:
        """打印网络信息。
        :param net: 待打印的网络
        :return: None
        """
        if trainer.runtime_cfg['print_net']:
            input_size = trainer.input_size
            if isinstance(input_size, Iterable):
                try:
                    summary(net, input_size=(trainer.hps['batch_size'], *input_size), device=net.device)
                except Exception as e:
                    msg = f"打印网络时遇到错误：{e}，只显示网络结构！"
                    warnings.warn(msg)
                    print(net)
            else:
                warnings.warn(f"输入形状{input_size}无法解析，只显示网络结构！")
                print(net)


class prepare:
    """函数准备装饰器，进行不同类型的准备，请使用@prepare()进行调用"""

    # 是否处于k_fold模式
    k_fold = False

    def __init__(self, what='train', mute=False):
        """函数准备装饰器，进行不同类型的准备，请使用@prepare()进行调用

        'train': 需要传入参数net、prepare_args、criterion_a、*args，其含义分别为训练待使用的网络对象，
        准备动作所需位置参数，训练所需评价指标函数，训练函数其他位置参数。训练后会将优化器、学习率规划器以及损失函数销毁

        'valid': 需要传入参数net、pbar、criterion_a、*args，其含义分别为验证待使用的网络对象，
        验证所用进度条，验证所需评价指标函数，验证函数其他位置参数

        'test': 需要传入参数net、criterion_a、ls_fn_args、*args，其含义分别为测试待使用的网络对象，
        测试所需评价指标函数，测试所用损失函数参数，测试函数其他位置参数

        :param what: 进行何种准备
        """
        self.what = what
        self.mute = mute

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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
            elif self.what == 'predict':
                args = self.__prepare_predict(*args)
                # 执行本体函数
                with torch.no_grad():
                    result = func(*args, **kwargs)
                self.__after_predict(args[0])
            else:
                raise ValueError(f'未知准备类型{self.what}!')
            return result

        return wrapper

    @net_builder()
    def __prepare_train(self, *args):
        """进行训练准备
        根据需要进行网络对象创建和训练准备工作。

        :param args: 实际签名为：
            def __prepare_train(
                trainer: Trainer,
                module_class: type, __m_init_args: tuple, __m_init_kwargs: dict,
                prepare_args: tuple, *args
            ) -> History
            module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
            不需要创建网络则rainer.module不能为None，此时三个参数无需传递；
            prepare_args用于进行网络训练准备。如不需要则trainer.module.ready需为True，此时无需传递prepare_args
        :return: trainer, *args
        """
        trainer, net, *args = args
        # if not net.ready:
        #     # 训练前准备
        #     prepare_args = trainer.prepare_args
        #     assert len(prepare_args) == len(inspect.signature(net.prepare_training).parameters), \
        #         (f"网络训练需要按照顺序提供优化器参数、学习率规划器参数、训练损失函数参数、测试损失函数参数；"
        #          f"本次训练只收到了{len(prepare_args)}个参数")
        #     net.prepare_training(*prepare_args)
        configure_network(net, True, *trainer.prepare_args)
        # 进度条更新
        if hasattr(trainer, 'pbar'):
            trainer.pbar.set_description('训练中……')
        args = trainer, *args
        return args

    def __after_train(self, trainer):
        """训练后处理
        消除网络的训练痕迹，包括优化器、学习率规划器、损失函数、各式名称以及ready标志。
        如果prepare对象为k_fold模式，则什么也不做

        :param trainer: 训练器对象，获取其中的module对象
        :return: None
        """
        if self.k_fold:
            return
        # net = trainer.module
        # # 清除训练痕迹
        # del net.optimizer_s, net.scheduler_s, net.lr_names, net.train_ls_fn_s, \
        #     net.train_ls_names, net.test_ls_fn_s, net.test_ls_names
        # net.ready = False
        trainer.module.deactivate()

    @staticmethod
    def __prepare_valid(*args):
        """进行验证准备
        设置神经网络模式以及更新进度条

        :param args: 需要传递的第一个参数为trainer对象
        :return: args
        """
        trainer, *args = args
        trainer.module.eval()
        if hasattr(trainer, 'pbar'):
            trainer.pbar.set_description('验证中……')
        return trainer, *args

    @staticmethod
    def __prepare_test(*args):
        """进行测试准备
        进行测试准备，设置神经网络模式以及更新进度条

        :param args: 需要传递的第一个参数为trainer对象
        :return: args
        """
        trainer, *args = args
        # net = trainer.module
        # # 进行测试准备，获取损失函数
        # with warnings.catch_warnings():
        #     # 忽视优化器参数不足警告
        #     warnings.simplefilter('ignore', category=UserWarning)
        #     ts_ls_args = trainer.prepare_args[3]
        #     if isinstance(ts_ls_args[0], list):
        #         o_args, l_args, tr_ls_args = [[]], [[]], [[]]
        #     elif isinstance(ts_ls_args[0], tuple):
        #         o_args, l_args, tr_ls_args = [()], [()], [()]
        #     net.prepare_training(o_args, l_args, tr_ls_args, ts_ls_args)
        # del net.optimizer_s, net.scheduler_s, net.train_ls_fn_s, net.train_ls_names
        # # 设置神经网络模式
        # net.eval()
        configure_network(trainer.module, False, ts_ls_args=trainer.prepare_args[3])
        return trainer, *args

    @staticmethod
    def __after_test(trainer):
        """测试后处理
        删除测试痕迹

        :param trainer: 训练器对象，神经网络对象的来源
        :return: None
        """
        # net = trainer.module
        # # 清除测试痕迹
        # del net.lr_names, net.test_ls_fn_s, net.test_ls_names
        trainer.module.deactivate()

    @net_builder()
    def __prepare_predict(self, *args):
        """进行预测准备
        进行预测准备，设置神经网络模式以及更新进度条

        :param args: 需要传递的第一个参数为trainer对象
        :return: args
        """
        trainer, net, data_iter, *args = args
        # if not net.ready:
        #     # 预处理损失函数参数
        #     ls_fn_args = trainer.prepare_args[3]
        #     for (_, kwargs) in ls_fn_args:
        #         kwargs['size_averaged'] = False
        #     # 进行测试准备，获取损失函数
        #     with warnings.catch_warnings():
        #         # 忽视优化器参数不足警告
        #         warnings.simplefilter('ignore', category=UserWarning)
        #         net.prepare_training(ts_ls_args=ls_fn_args)
        #     del net.optimizer_s, net.scheduler_s, net.train_ls_fn_s, net.train_ls_names
        # 创建进度条
        trainer.pbar = tqdm(
            data_iter, unit='批', position=0, desc=f'正在计算结果……',
            mininterval=1, ncols=80
        )
        # # 设置神经网络模式
        # net.eval()
        configure_network(trainer.module, False, ts_ls_args=trainer.prepare_args[3])
        return trainer, *args

    @staticmethod
    def __after_predict(trainer):
        """预测后处理
        清除训练器携带的网络以及进度条

        :param trainer: 训练器对象，神经网络对象的来源
        :return: None
        """
        trainer.pbar.close()
        del trainer.module
