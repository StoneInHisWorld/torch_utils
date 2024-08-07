import json

import pandas as pd
import torch

from utils.experiment import Experiment
from utils.func import pytools


def init_log(path):
    """日志初始化方法
    日志初始化只会增加一个条目，该条目里只有“exp_no==1”的信息
    :param path:
    :return:
    """
    with open(path, 'w', encoding='utf-8') as log:
        log.write("exp_no\n1\n")


class ControlPanel:
    """控制台类负责读取、管理动态运行参数、超参数组合，以及实验对象的提供"""

    def __init__(self, datasource,
                 hp_cfg_path: str,
                 runtime_cfg_path: str,
                 log_path: str = None,
                 net_path: str = None,
                 plot_path: str = None):
        """控制台类。
        负责读取、管理动态运行参数、超参数组合，以及实验对象的提供。
        迭代每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变，每次迭代后会更新运行配置参数。

        :param datasource: 训练数据来源
        :param hp_cfg_path: 超参数配置文件路径
        :param runtime_cfg_path: 运行配置文件路径
        :param log_path: 日志文件存储路径
        :param net_path: 网络文件存储路径
        """
        # 路径检查以及路径提取
        pytools.check_path(hp_cfg_path)
        pytools.check_path(runtime_cfg_path)
        if log_path is not None:
            pytools.check_path(log_path, init_log)
        if net_path is not None:
            pytools.check_path(net_path)
        if plot_path is not None:
            pytools.check_path(plot_path)
        # 路径整理
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
        torch.random.manual_seed(self['random_seed'])
        # # 读取实验编号
        self.__read_expno()

    def __read_expno(self):
        """读取实验编号
        从日志中读取最后一组实验数据的实验编号，从而推算出即将进行的所有实验组对应的编号。
        本函数将会创建self.exp_no以及self.last_expno属性。
        :return: None
        """
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
        # 计算总共需要进行的实验组数
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            n_exp = 1
            for v in hyper_params.values():
                n_exp *= len(v)
        # 最后一组实验的实验编号
        self.last_expno = self.exp_no + n_exp - 1

    def __iter__(self):
        """迭代
        每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变。
        每次迭代会更新运行配置参数。
        :return: None
        """
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            for hps in pytools.permutation([], *hyper_params.values()):
                hyper_params = {k: v for k, v in zip(hyper_params.keys(), hps)}
                # 构造实验对象
                self.__cur_exp = Experiment(
                    self.exp_no, self.__datasource,
                    hyper_params, self.cfg_dict,
                    self.__pp, self.__lp, self.__np
                )
                print(
                    f'\r---------------------------'
                    f'实验{self.exp_no}号/{self.last_expno}号'
                    f'---------------------------'
                )
                yield self.__cur_exp
                self.__read_runtime_cfg()

    def __getitem__(self, item):
        """获取控制面板中的运行配置参数。

        :param item: 运行配置参数名称
        :return: 运行配置参数值
        """
        assert item in self.cfg_dict.keys(), f'设置文件中不存在{item}参数！'
        return self.cfg_dict[item]

    def __read_runtime_cfg(self):
        """读取运行配置并更新实验编号"""
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = json.load(config)
            assert config_dict.keys() == self.cfg_dict.keys(), '在运行期间，不允许添加新的运行设置参数！'
            for k, v in config_dict.items():
                self.cfg_dict[k] = v
        # 更新实验编号
        self.exp_no += 1

    # def __list_net(self, net, input_size, batch_size) -> None:
    #     """
    #     打印网络信息。
    #     :param net: 待打印的网络信息。
    #     :param input_size: 网络输入参数。
    #     :param batch_size: 训练的批量大小。
    #     :return: None
    #     """
    #     if self['print_net']:
    #         try:
    #             summary(net, input_size=input_size, batch_size=batch_size)
    #         except RuntimeError as _:
    #             print(net)

    # def __plot_history(self, history, **plot_kwargs) -> None:
    #     # 检查参数设置
    #     cfg_range = ['plot', 'save', 'no']
    #     cfg = self['plot_history']
    #     if not pytools.check_para('plot_history', cfg, cfg_range):
    #         print('请检查setting.json中参数plot_history设置是否正确，本次不予绘制历史趋势图！')
    #         return
    #     if cfg == 'no':
    #         return
    #     if self.__pp is None:
    #         warnings.warn('未指定绘图路径，不予保存历史趋势图！')
    #     savefig_as = None if self.__pp is None or cfg == 'plot' else self.__pp + str(self.exp_no) + '.jpg'
    #     # 绘图
    #     ltools.plot_history(
    #         history, mute=self['plot_mute'],
    #         title='EXP NO.' + str(self.exp_no),
    #         savefig_as=savefig_as, **plot_kwargs
    #     )
    #     print('已绘制历史趋势图')

    # def register_result(self, history, test_log=None, **plot_kwargs) -> None:
    #     """根据训练历史记录进行输出，并进行日志参数的记录。
    #     在神经网络训练完成后，需要调用本函数将结果注册到超参数控制台。
    #     :param history: 训练历史记录
    #     :param test_log: 测试记录
    #     :return: None
    #     """
    #     log_msg = {}
    #     if len(history) <= 0:
    #         raise ValueError('历史记录对象为空！')
    #     # 输出训练部分的数据
    #     for name, log in history:
    #         if name != name.replace('train_', '训练'):
    #             # 输出训练信息，并记录
    #             print(f"{name.replace('train_', '训练')} = {log[-1]:.5f},", end=' ')
    #             log_msg[name] = log[-1]
    #     # 输出验证部分的数据
    #     print('\b\b')
    #     for name, log in history:
    #         if name != name.replace('valid_', '验证'):
    #             # 输出验证信息，并记录
    #             print(f"{name.replace('valid_', '验证')} = {log[-1]:.5f},", end=' ')
    #             log_msg[name] = log[-1]
    #     # 输出测试部分的数据
    #     print('\b\b')
    #     if test_log is not None:
    #         for k, v in test_log.items():
    #             print(f"{k.replace('test_', '测试')} = {v:.5f},", end=' ')
    #         print('\b\b')
    #     log_msg.update(test_log)
    #     self.__plot_history(
    #         history, **plot_kwargs
    #     )
    #     self.__cur_exp.add_logMsg(
    #         True, **log_msg, data_portion=self['data_portion']
    #     )
    #     if self.device != torch.device('cpu') and self["cuda_memrecord"]:
    #         self.__cur_exp.add_logMsg(
    #             True,
    #             max_GPUmemory_allocated=torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
    #             max_GPUmemory_reserved=torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
    #         )
    #         torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def runtime_cfg(self):
        return self.cfg_dict

    @property
    def device(self):
        return torch.device(self['device'])


