import json
import warnings

import pandas as pd
import torch
from torchsummary import summary

import utils.func.log_tools as ltools
from utils.func import pytools
from utils.trainer import Trainer


def init_log(path):
    with open(path, 'w', encoding='utf-8') as log:
        log.write("exp_no\n1\n")


class ControlPanel:

    def __init__(self, datasource,
                 hp_cfg_path: str,
                 runtime_cfg_path: str,
                 log_path: str = None,
                 net_path: str = None,
                 plot_path: str = None):
        """
        控制台类。用于读取运行参数设置，设定训练超参数以及自动编写日志文件等一系列与网络构建无关的操作。
        :param datasource: 训练数据来源
        :param hp_cfg_path: 超参数配置文件路径
        :param runtime_cfg_path: 运行配置文件路径
        :param log_path: 日志文件存储路径
        :param net_path: 网络文件存储路径
        """
        pytools.check_path(hp_cfg_path)
        pytools.check_path(runtime_cfg_path)
        if log_path is not None:
            pytools.check_path(log_path, init_log)
        if net_path is not None:
            pytools.check_path(net_path)
        if plot_path is not None:
            pytools.check_path(plot_path)
        self.__rcp = runtime_cfg_path
        self.__hcp = hp_cfg_path
        self.__lp = log_path
        self.__np = net_path
        self.__pp = plot_path
        self.__datasource = datasource
        # 读取运行配置
        with open(self.__rcp, 'r') as cfg:
            self.cfg_dict = json.load(cfg)
        # 设置随机种子
        self.random_seed = self['random_seed']
        torch.random.manual_seed(self.random_seed)
        # # 读取实验编号
        self.__read_expno()

    def __read_expno(self):
        # 读取实验编号
        if self.__lp is not None:
            try:
                log = pd.read_csv(self.__lp)
                exp_no = log.iloc[-1]['exp_no'] + 1
            except Exception as _:
                exp_no = 1
        else:
            exp_no = 1
        assert exp_no > 0, f'训练序号需为正整数，但读取到的序号为{exp_no}'
        self.exp_no = int(exp_no)
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            n_exp = 1
            for v in hyper_params.values():
                n_exp *= len(v)
        self.last_expno = self.exp_no + n_exp - 1

    def __iter__(self):
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            for hps in pytools.permutation([], *hyper_params.values()):
                hyper_params = {k: v for k, v in zip(hyper_params.keys(), hps)}
                self.__cur_trainer = Trainer(
                    self.__datasource, hyper_params, self.exp_no,
                    self.__lp, self.__np, self['print_net'], self['save_net']
                )
                print(
                    f'\r---------------------------实验{self.exp_no}号/{self.last_expno}号'
                    f'---------------------------'
                )
                yield self.__cur_trainer
                self.__read_runtime_cfg()

    def __getitem__(self, item):
        """
        获取控制面板中的运行配置参数。
        :param item: 运行配置参数名称
        :return: 运行配置参数值
        """
        assert item in self.cfg_dict.keys(), f'设置文件中不存在{item}参数！'
        return self.cfg_dict[item]

    def __read_runtime_cfg(self):
        """
        读取运行配置，在每组超参数训练前都会进行本操作。
        :return: None
        """
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = json.load(config)
            assert config_dict.keys() == self.cfg_dict.keys(), '在运行期间，不允许添加新的运行设置参数！'
            for k, v in config_dict.items():
                self.cfg_dict[k] = v
        # 更新实验编号
        self.exp_no += 1

    def __list_net(self, net, input_size, batch_size) -> None:
        """
        打印网络信息。
        :param net: 待打印的网络信息。
        :param input_size: 网络输入参数。
        :param batch_size: 训练的批量大小。
        :return: None
        """
        if self['print_net']:
            try:
                summary(net, input_size=input_size, batch_size=batch_size)
            except RuntimeError as _:
                print(net)

    # def __plot_history(self, history, cfg, mute, ls_fn, acc_fn) -> None:
    def __plot_history(self, history, **plot_kwargs) -> None:
        # 检查参数设置
        cfg_range = ['plot', 'save', 'no']
        cfg = self['plot_history']
        if not pytools.check_para('plot_history', cfg, cfg_range):
            print('请检查setting.json中参数plot_history设置是否正确，本次不予绘制历史趋势图！')
            return
        if cfg == 'no':
            return
        if self.__pp is None:
            warnings.warn('未指定绘图路径，不予保存历史趋势图！')
        savefig_as = None if self.__pp is None or cfg == 'plot' else self.__pp + str(self.exp_no) + '.jpg'
        # 绘图
        ltools.plot_history(
            history, mute=self['plot_mute'],
            title='EXP NO.' + str(self.exp_no),
            savefig_as=savefig_as, **plot_kwargs
        )
        print('已绘制历史趋势图')

    # def register_result(self, history, test_acc=None, test_ls=None,
    #                     ls_fn=None, acc_fn=None) -> None:
    def register_result(self, history, test_log=None, **plot_kwargs) -> None:
        """根据训练历史记录进行输出，并进行日志参数的记录。
        在神经网络训练完成后，需要调用本函数将结果注册到超参数控制台。
        :param history: 训练历史记录
        :param test_log: 测试记录
        :return: None
        """
        log_msg = {}
        # 输出训练部分的数据
        for name, log in history:
            if name != name.replace('train_', '训练'):
                # 输出训练信息，并记录
                print(f"{name.replace('train_', '训练')} = {log[-1]:.5f},", end=' ')
                log_msg[name] = log[-1]
        # 输出验证部分的数据
        print('\b\b')
        for name, log in history:
            if name != name.replace('valid_', '验证'):
                # 输出验证信息，并记录
                print(f"{name.replace('valid_', '验证')} = {log[-1]:.5f},", end=' ')
                log_msg[name] = log[-1]
        print('\b\b')
        if test_log is not None:
            for k, v in test_log.items():
                print(f"{k.replace('test_', '测试')} = {v:.5f},", end=' ')
            print('\b\b')
        log_msg.update(test_log)
        self.__plot_history(
            history, **plot_kwargs
        )
        self.__cur_trainer.add_logMsg(
            True, **log_msg, data_portion=self['data_portion']
        )

    @property
    def device(self):
        return torch.device(self['device'])


