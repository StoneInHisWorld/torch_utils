import json

import pandas as pd
import torch
from torchsummary import summary

from utils import tools
from utils.tools import permutation
from utils.trainer import Trainer


class ControlPanel:

    def __init__(self, datasource,
                 hp_config_path: str,
                 runtime_config_path: str,
                 log_path: str = None,
                 net_path: str = None):
        """
        控制台类。用于读取运行参数设置，设定训练超参数以及自动编写日志文件等一系列与网络构建无关的操作。
        :param datasource: 训练数据来源
        :param hp_config_path: 超参数配置文件路径
        :param runtime_config_path: 运行配置文件路径
        :param log_path: 日志文件存储路径
        :param net_path: 网络文件存储路径
        """

        def init_log(path):
            with open(path, 'w', encoding='utf-8') as log:
                log.write("exp_no\n1\n")

        tools.check_path(hp_config_path)
        tools.check_path(runtime_config_path)
        tools.check_path(log_path, init_log)
        tools.check_path(net_path)
        self.__rcp = runtime_config_path
        self.__hcp = hp_config_path
        self.__lp = log_path
        self.__np = net_path
        self.__datasource = datasource.__class__.__name__
        # 读取运行配置
        with open(self.__rcp, 'r') as config:
            self.config_dict = json.load(config)
        # 设置随机种子
        self.random_seed = self['random_seed']
        torch.random.manual_seed(self.random_seed)
        # 读取实验编号
        if self.__lp is not None:
            try:
                log = pd.read_csv(self.__lp)
                self.exp_no = log.iloc[-1]['exp_no'] + 1
                # self.config_dict['exp_no'] = log.iloc[-1]['exp_no'] + 1
            except Exception as _:
                self.exp_no = 1
                # self.config_dict['exp_no'] = 1

    def __iter__(self):
        with open(self.__hcp, 'r', encoding='utf-8') as config:
            hyper_params = json.load(config)
            for hps in permutation([], *hyper_params.values()):
                hyper_params = {k: v for k, v in zip(hyper_params.keys(), hps)}
                # yield Trainer(
                #     self.__datasource.__class__.__name__, hyper_params, self.exp_no,
                #     self.__lp, self.__np
                # )
                yield Trainer(
                    self.__datasource, hyper_params, self.exp_no,
                    self.__lp, self.__np, self['save_net']
                )
                self.__read_running_config()

    def __getitem__(self, item):
        """
        获取控制面板中的运行配置参数。
        :param item: 运行配置参数名称
        :return: 运行配置参数值
        """
        assert item in self.config_dict.keys(), f'设置文件中不存在{item}参数！'
        return self.config_dict[item]

    def __read_running_config(self):
        """
        读取运行配置，在每组超参数训练前都会进行本操作。
        :return: None
        """
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = json.load(config)
            assert config_dict.keys() == self.config_dict.keys(), '在运行期间，不允许添加新的运行设置参数！'
            for k, v in config_dict.items():
                self.config_dict[k] = v
        # 更新实验编号
        self.exp_no += 1

    # TODO：移入Trainer中，并将其从main.py中隐藏
    def list_net(self, net, input_size, batch_size):
        """
        是否打印网络信息
        :param net:
        :param input_size:
        :param batch_size:
        :return:
        """
        # assert hasattr(self, "print_net"), '设置文件中不存在"print_net"参数！'
        # if self.print_net:
        #     try:
        #         summary(net, input_size=input_size, batch_size=batch_size)
        #     except RuntimeError as _:
        #         print(net)
        if self['print_net']:
            try:
                summary(net, input_size=input_size, batch_size=batch_size)
            except RuntimeError as _:
                print(net)

    def plot_history(self, history, xlabel='num_epochs', ylabel='loss', title=None, save_path=None):
        # assert hasattr(self, 'pic_mute'), '配置文件中缺少参数"pic_mute"'
        # assert hasattr(self, 'plot'), '配置文件中缺少参数"plot"'
        # if self.plot:
        #     print('plotting...')
        #     tools.plot_history(
        #         history, xlabel=xlabel, ylabel=ylabel, mute=self.pic_mute, title=title,
        #         savefig_as=save_path
        #     )
        if self['plot']:
            print('plotting...')
            tools.plot_history(
                history, xlabel=xlabel, ylabel=ylabel, mute=self['pic_mute'], title=title,
                savefig_as=save_path
            )

    @property
    def device(self):
        return torch.device(self['device'])


