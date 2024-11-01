import json
import os.path

import pandas as pd
import torch

from config.init_cfg import init_log, init_settings, init_hps
# from utils.experiment import Experiment
# from utils.func import pytools
from .experiment import Experiment
from .func import pytools as ptools
# from utils import Experiment, ptools


class ControlPanel:
    """控制台类负责读取、管理动态运行参数、超参数组合，以及实验对象的提供"""

    def __init__(self, datasource, net_type, cfg_root):
        """控制面板

        负责读取、管理动态运行参数、超参数组合，生成超参数配置文件路径、日志文件存储路径以及网络文件存储目录，提供实验Experiment对象。
        迭代每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变，每次迭代后会更新运行配置参数。
        :param datasource: 训练数据来源
        :param net_type: 训练所用模型类
        :param cfg_root: 运行配置文件路径
        """
        # 生成运行动态配置
        self.__rcp = os.path.join(cfg_root, f'settings.json')  # 运行配置json文件路径
        # 读取运行配置
        ptools.check_path(self.__rcp, init_settings)
        self.__read_runtime_cfg()
        # 生成其他路径
        net_name = net_type.__name__.lower()
        self.__hcp = os.path.join(cfg_root, f'hp_control/{net_name}_hp.json')  # 网络训练超参数文件路径
        log_root = self.cfg_dict['log_root']
        self.__lp = os.path.join(log_root, f'{net_name}_log.csv')  # 日志文件存储路径
        self.__np = os.path.join(log_root, f'trained_net/{net_name}/')  # 训练成果网络存储路径
        self.__pp = os.path.join(log_root, f'imgs/{net_name}/')  # 历史趋势图存储路径
        # 路径检查
        ptools.check_path(self.__hcp, init_hps)
        ptools.check_path(self.__lp, init_log)
        ptools.check_path(self.__np)
        ptools.check_path(self.__pp)
        # 设置随机种子
        torch.random.manual_seed(self['random_seed'])
        # # 读取实验编号
        self.__read_expno()
        self.__datasource = datasource

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
            for hps in ptools.permutation([], *hyper_params.values()):
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
                # 读取运行动态参数
                self.__read_runtime_cfg()
                # 更新实验编号
                self.exp_no += 1

    def __getitem__(self, item):
        """获取控制面板中的运行配置参数。

        :param item: 运行配置参数名称
        :return: 运行配置参数值
        """
        assert item in self.cfg_dict.keys(), f'设置文件中不存在{item}参数！'
        return self.cfg_dict[item]

    def __read_runtime_cfg(self):
        """读取运行配置并更新或者赋值"""
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = json.load(config)
            if hasattr(self, 'cfg_dict'):
                # 更新运行动态参数
                assert config_dict.keys() == self.cfg_dict.keys(), '在运行期间，不允许添加新的运行设置参数！'
                for k, v in config_dict.items():
                    self.cfg_dict[k] = v
            else:
                # 首次获取运行动态参数
                self.cfg_dict = config_dict

    @property
    def runtime_cfg(self):
        return self.cfg_dict

    @property
    def device(self):
        return torch.device(self['device'])
