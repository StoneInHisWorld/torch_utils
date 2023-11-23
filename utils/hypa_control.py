import json
import time

import pandas as pd
import torch
from torchsummary import summary

from utils import tools
from utils.tools import permutation


class ControlPanel:

    def __init__(self, datasource,
                 hypa_config_path: str,
                 runtime_config_path: str,
                 log_path: str = None,
                 net_path: str = None):
        """
        控制台类。用于读取运行参数设置，设定训练超参数以及自动编写日志文件等一系列与网络构建无关的操作。
        """

        def init_log(path):
            with open(path, 'w', encoding='utf-8') as log:
                log.write("exp_no\n1\n")

        tools.check_path(hypa_config_path)
        tools.check_path(runtime_config_path)
        tools.check_path(log_path, init_log)
        # TODO：添加网络保存路径的检查
        self.__rcp = runtime_config_path
        self.__hcp = hypa_config_path
        self.__lp = log_path
        self.__np = net_path
        self.__datasource = datasource
        self.__extra_lm = {}
        # 读取运行配置
        with open(self.__rcp, 'r') as config:
            config_dict = json.load(config)
            self.__rck = config_dict.keys()
            for k, v in config_dict.items():
                setattr(self, k, v)
        # 读取实验编号
        if self.__lp is not None:
            try:
                log = pd.read_csv(self.__lp)
                self.exp_no = log.iloc[-1]['exp_no'] + 1
            except:
                self.exp_no = 1

    def __iter__(self):
        with open(self.__hcp, 'r', encoding='utf-8') as config:
            hyper_params = json.load(config)
            for hps in permutation([], *hyper_params.values()):
                hyper_params = {k: v for k, v in zip(hyper_params.keys(), hps)}
                yield Trainer(self.__datasource.__class__.__name__, hyper_params, self.__lp, self.__np)
                self.__read_running_config()

    def __read_running_config(self):
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = json.load(config)
            assert config_dict.keys() == self.__rck, '在运行期间，不允许添加新的运行设置参数！'
            for k, v in config_dict.items():
                setattr(self, k, v)
        # 更新实验编号
        self.exp_no += 1

    def list_net(self, net, input_size, batch_size):
        assert hasattr(self, "print_net"), '设置文件中不存在"print_net"参数！'
        if self.print_net:
            try:
                summary(net, input_size=input_size, batch_size=batch_size)
            except RuntimeError as _:
                print(net)

    def plot_history(self, history, xlabel='num_epochs', ylabel='loss', title=None, save_path=None):
        assert hasattr(self, 'pic_mute'), '配置文件中缺少参数"pic_mute"'
        assert hasattr(self, 'plot'), '配置文件中缺少参数"plot"'
        if self.plot:
            print('plotting...')
            tools.plot_history(
                history, xlabel=xlabel, ylabel=ylabel, mute=self.pic_mute, title=title,
                savefig_as=save_path
            )

    @property
    def running_device(self):
        assert hasattr(self, "device"), '设置文件中不存在"device"参数！'
        return torch.device(self.device)

    @property
    def running_randomseed(self):
        assert hasattr(self, "random_seed"), '设置文件中不存在"random_seed"参数！'
        return self.random_seed

    @property
    def running_dataportion(self):
        assert hasattr(self, "data_portion"), '设置文件中不存在"data_portion"参数！'
        return self.data_portion

    @property
    def running_expno(self):
        assert hasattr(self, "exp_no"), '设置文件中不存在"exp_no"参数！'
        return self.exp_no


class Trainer:

    def __init__(self, datasource, hyper_parameters: dict,
                 log_path: str = None,
                 net_path: str = None):
        """
        训练器。
        使用with上下文管理器以充分利用其全部功能。
        with启用时，训练器会计时以记录本次训练所花时间。
        with退出时，会编写日志记录本次训练的数据。
        可以向日志文件中加入额外参数。
        可以对网络进行持久化。
        """
        self.__extra_lm = {}
        self.__hp = hyper_parameters
        self.__lp = log_path
        self.__np = net_path
        self.datasource = datasource

    def __enter__(self):
        self.start = time.time()
        return self.__hp.values()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f'exc_type: {exc_type}')
            print(f'exc_val: {exc_val}')
        time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start))
        self.__hp.update({'exc_val': exc_val, "duration": time_span, "dataset": self.datasource})
        if self.__lp is not None and exc_type != KeyboardInterrupt:
            self.__write_log(**self.__hp)

    def __write_log(self, **kwargs):
        print('logging...')
        kwargs.update(self.__extra_lm)
        tools.write_log(self.__lp, **kwargs)

    def add_logMsg(self, mute=True, **kwargs):
        self.__extra_lm = kwargs
        if not mute:
            print(self.__extra_lm)

    def save_net(self, net: torch.nn.Module, exp_no: int, entire=False):
        if self.__np is None:
            print("未指定模型保存路径，不予保存模型！")
        if entire:
            torch.save(net, self.__np + f'{exp_no}.ptm')
        else:
            torch.save(net.state_dict(), self.__np + f'{exp_no}.ptsd')
