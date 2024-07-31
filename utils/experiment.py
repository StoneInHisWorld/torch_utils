import time
import warnings

import torch

import utils.func.log_tools as ltools
from utils.func import pytools


class Experiment:
    """实验对象负责神经网络训练的相关周边操作，计时、显存监控、日志编写、网络持久化、历史趋势图绘制及保存"""

    sn_range = ['no', 'entire', 'state']

    def __init__(self,
                 exp_no: int, datasource: type,
                 hyper_parameters: dict, runtime_cfg: dict,
                 plot_path: str = None, log_path: str = None, net_path: str = None
                 ):
        """实验对象
        进行神经网络训练的相关周边操作，计时、显存监控、日志编写、网络持久化、历史趋势图绘制及保存。
        请使用上下文管理器来调用本对象以启用全部功能，示例：`with Experiment() as hps:`。使用上下文管理器后，能够自动开启计时以及显存监控功能，
        在退出时会自动进行日志编写、异常信息处理以及网络持久化。
        在神经网络训练完成后，需要调用self.register_result将结果注册，并对其进行一系列输出操作。
        可以调用self.add_logMsg来增加日志条目信息。

        :param exp_no: 实验编号
        :param datasource: 实验所用数据源
        :param hyper_parameters: 超参数组合
        :param runtime_cfg: 运行动态参数
        :param log_path: 日志所在路径
        :param net_path: 网络保存路径
        """
        self.__extra_lm = {}
        self.__hp = hyper_parameters
        self.__pp = plot_path
        self.__lp = log_path
        self.__np = net_path
        self.__exp_no = exp_no
        self.__runtime_cfg = runtime_cfg
        self.datasource = datasource
        self.__net = None

    def __enter__(self):
        """训练对象的上下文管理进入方法
        负责进行计时、超参数打印以及显存监控。
        """
        # 计时开始
        self.start = time.time()
        # 打印本次训练超参数
        for k, v in self.__hp.items():
            print(k + ': ' + str(v))
        print(
            '\r----------------------------------------------------------------'
        )
        device = self.__runtime_cfg['device']
        cuda_memrecord = self.__runtime_cfg['cuda_memrecord']
        # 开启显存监控
        if device != 'cpu':
            torch.cuda.memory._record_memory_history(cuda_memrecord)
        elif device == 'cpu' and cuda_memrecord:
            warnings.warn(
                f'运行设备为{device}，不支持显存监控！请使用支持CUDA的处理机，或者设置cuda_memrecord为false')

        return self.__hp

    def __exit__(self, exc_type, exc_val, exc_tb):
        """训练器对象的退出动作。
        进行日志编写以及网络保存操作。

        :param exc_type: 出现的异常类型
        :param exc_val: 出现的异常值
        :param exc_tb: 异常的路径回溯
        """
        # 进行日志编写
        if exc_type is not None:
            if exc_type != KeyboardInterrupt:
                # 出现异常则记录
                print(f'exc_type: {exc_type}')
                print(f'exc_val: {exc_val}')
                self.__hp.update({'exc_val': exc_val})
            else:
                # 键盘中断则什么也不做
                print('该组超参数实验被中断！')
                for i in range(10, 0, -1):
                    print(f'\r将在{i}秒后继续进行下一组超参数实验，再中断一次即可终止整个程序', end='', flush=True)
                    time.sleep(1)
                return True
                # raise KeyboardInterrupt('该组超参数实验被中断！')
        # 记录时间信息
        time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start))
        self.__hp.update({
            'exp_no': self.__exp_no, "duration": time_span, "dataset": self.datasource.__name__
        })
        if self.__lp is not None:
            # 指定了日志路径，则进行日志记录
            self.__write_log(**self.__hp)
        # 保存训练生成的网络
        self.__save_net()

    def register_result(self, net, history, test_log=None, **plot_kwargs) -> None:
        """根据训练历史记录进行输出，并进行日志参数的记录。
        在神经网络训练完成后，需要调用本函数将结果注册到超参数控制台。
        :param net: 训练完成的网络对象
        :param history: 训练历史记录
        :param test_log: 测试记录
        :return: None
        """
        self.__net = net
        # 绘制历史趋势图
        log_msg = self.__print_result(history, test_log)
        with warnings.catch_warnings():
            # 忽视空图图例警告
            warnings.simplefilter('ignore', category=UserWarning)
            self.__plot_history(history, **plot_kwargs)
        # 编辑日志条目
        self.add_logMsg(
            True, **log_msg, data_portion=self.__runtime_cfg['data_portion']
        )
        if self.device != torch.device('cpu') and self.__runtime_cfg["cuda_memrecord"]:
            self.add_logMsg(
                True,
                max_GPUmemory_allocated=torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
                max_GPUmemory_reserved=torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    def __print_result(self, history, test_log):
        log_msg = {}
        if len(history) <= 0:
            raise ValueError('历史记录对象为空！')
        # 输出训练部分的数据
        train_history = filter(lambda h: h[0].startswith('train_'), history)
        valid_history = list(filter(lambda h: h[0].startswith('valid_'), history))
        for name, log in train_history:
            # 输出训练信息，并记录
            print(f"{name.replace('train_', '训练')} = {log[-1]:.5f},", end=' ')
            log_msg[name] = log[-1]
        # 输出验证部分的数据
        if len(valid_history) > 0:
            print('\b\b')
        for name, log in valid_history:
            # 输出验证信息，并记录
            print(f"{name.replace('valid_', '验证')} = {log[-1]:.5f},", end=' ')
            log_msg[name] = log[-1]
        # 输出测试部分的数据
        print('\b\b')
        if test_log is not None:
            for k, v in test_log.items():
                print(f"{k.replace('test_', '测试')} = {v:.5f},", end=' ')
            print('\b\b')
        log_msg.update(test_log)
        return log_msg

    def add_logMsg(self, mute=True, **kwargs):
        """增加日志条目信息

        :param mute: 是否打印日志条目所有信息
        :param kwargs: 增加日志条目信息
        :return: None
        """
        self.__extra_lm.update(kwargs)
        if not mute:
            print(self.__extra_lm)

    def __write_log(self, **kwargs):
        """日志编写函数
        :param kwargs: 日志条目内容
        :return: None
        """
        kwargs.update(self.__extra_lm)
        ltools.write_log(self.__lp, **kwargs)
        print('已编写日志')

    def __save_net(self) -> None:
        """保存实验对象持有网络
        根据动态运行参数进行相应的网络保存动作，具有三种保存模式，保存模式由动态运行参数save_net指定：
        entire：指持久化整个网络对象
        state：指持久化网络对象参数
        no：指不进行持久化

        :return: None
        """
        if self.__np is None:
            print("未指定模型保存路径，不予保存模型！")
            return
        if not isinstance(self.__net, torch.nn.Module):
            print('训练器对象未得到训练网络对象，因此不予保存网络！')
            return
        save_net = self.__runtime_cfg['save_net']
        if not pytools.check_para('save_net', save_net, self.sn_range):
            warnings.warn(
                '请检查setting.json中参数save_net设置是否正确，本次不予保存模型！',
                UserWarning
            )
            return
        if save_net == 'entire':
            torch.save(self.__net, self.__np + f'{self.__exp_no}.ptm')
        elif save_net == 'state':
            torch.save(self.__net.state_dict(), self.__np + f'{self.__exp_no}.ptsd')
        print('已保存网络')

    def __plot_history(self, history, **plot_kwargs) -> None:
        """绘制历史趋势图
        根据需要绘制历史趋势图，有三种模式可选，模式选择由动态运行参数plot_history指定：
        plot： 绘制图像，但不进行保存
        save：绘制图像且保存，需要指定保存路径self.__pp，保存图片名为(实验编号.jpg)
        no：不绘制历史趋势图

        :param history: 历史记录对象
        :param plot_kwargs: 历史趋势图绘制关键字参数
        :return: None
        """
        # 检查参数设置
        cfg_range = ['plot', 'save', 'no']
        cfg = self.__runtime_cfg['plot_history']
        if not pytools.check_para('plot_history', cfg, cfg_range):
            warnings.warn(
                '请检查setting.json中参数plot_history设置是否正确，本次不予绘制历史趋势图！',
                UserWarning
            )
            return
        if cfg == 'no':
            return
        if self.__pp is None:
            warnings.warn('未指定绘图路径，不予保存历史趋势图！')
        savefig_as = None if self.__pp is None or cfg == 'plot' else (
                self.__pp + str(self.__exp_no) + '.jpg')
        # 绘图
        ltools.plot_history(
            history, mute=self.__runtime_cfg['plot_mute'],
            title='EXP NO.' + str(self.__exp_no),
            savefig_as=savefig_as, **plot_kwargs
        )
        print('已绘制历史趋势图')

    @property
    def device(self):
        return torch.device(self.__runtime_cfg['device'])
