import json
import warnings

import numpy as np
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

    def __iter__(self):
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            for hps in pytools.permutation([], *hyper_params.values()):
                hyper_params = {k: v for k, v in zip(hyper_params.keys(), hps)}
                self.__cur_trainer = Trainer(
                    self.__datasource, hyper_params, self.exp_no,
                    self.__lp, self.__np, self['print_net'], self['save_net']
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

    def __plot_history(self, history, cfg, mute, ls_fn, acc_fn) -> None:
        # 检查参数设置
        cfg_range = ['plot', 'save', 'no']
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
            history, mute=mute, ls_ylabel=ls_fn, acc_ylabel=acc_fn,
            title='EXP NO.' + str(self.exp_no), savefig_as=savefig_as
        )
        print('已绘制历史趋势图')

    def register_result(self, history, test_acc=None, test_ls=None,
                        ls_fn=None, acc_fn=None) -> None:
        train_acc, train_l = history["train_acc"][-1], history["train_l"][-1]
        try:
            valid_acc, valid_l = history["valid_acc"][-1], history["valid_l"][-1]
        except AttributeError as _:
            valid_acc, valid_l = np.nan, np.nan
        print(f'\r训练准确率 = {train_acc * 100:.3f}%, 训练损失 = {train_l:.5f}')
        print(f'验证准确率 = {valid_acc * 100:.3f}%, 验证损失 = {valid_l:.5f}')
        if test_acc is not None and test_ls is not None:
            print(f'测试准确率 = {test_acc * 100:.3f}%, 测试损失 = {test_ls:.5f}')
        self.__plot_history(
            history, self['plot_history'], self['plot_mute'], ls_fn, acc_fn
        )
        self.__cur_trainer.add_logMsg(
            True,
            train_l=train_l, train_acc=train_acc, valid_l=valid_l, valid_acc=valid_acc,
            test_acc=test_acc, test_ls=test_ls, data_portion=self['data_portion']
        )

    @property
    def device(self):
        return torch.device(self['device'])


