import json
import time

import torch
from torchsummary import summary

from utils import tools
from utils.tools import permutation


class ControlPanel:

    def __init__(self, hypa_config_path: str, runtime_config_path: str, datasource, log_path: str = None):
        self.__rcp = runtime_config_path
        self.__hcp = hypa_config_path
        self.__lp = log_path
        self.__datasource = datasource
        self.__extra_lm = {}

        with open(self.__rcp, 'r') as config:
            config_dict = json.load(config)
            self.__rck = config_dict.keys()
            for k, v in config_dict.items():
                setattr(self, k, v)
        # with open(hypa_config_path, 'r') as config:
        #     self.__hck = json.load(config).keys()
        #     for k, v in json.load(config):
        #         setattr(self, k, v)
        # with open(hypa_config_path, 'r', encoding='utf-8') as config:
        #     hypa_s_to_select = json.load(config).values()

    def __iter__(self):
        # TODO：提供超参数
        with open(self.__hcp, 'r', encoding='utf-8') as config:
            hyper_params = json.load(config)
            for hps in permutation([], *hyper_params.values()):
                hyper_params = {k: v for k, v in zip(hyper_params.keys(), hps)}
                yield Trainer(hyper_params, self.__lp)
                # self.__write_log(**dict.fromkeys(hyper_params.keys() + 'duration',
                #                                  hps.append(time_span)))
                self.__read_running_config()

    def __read_running_config(self):
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = json.load(config)
            assert config_dict.keys() == self.__rck, '在运行期间，不允许添加新的运行设置参数！'
            for k, v in config_dict.items():
                setattr(self, k, v)
            # self.data_portion, self.random_seed, self.pic_mute, self.print_net, self.device = \
            #     json.load(config).values()

    def list_net(self, net, input_size, batch_size):
        assert hasattr(self, "print_net"), '设置文件中不存在"print_net"参数！'
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
        return self.device


class Trainer:

    def __init__(self, hyper_parameters: dict, log_path: str = None):
        self.__extra_lm = {}
        self.__hp = hyper_parameters
        self.__lp = log_path

    def __enter__(self):
        self.start = time.time()
        return self.__hp.values()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f'exc_type: {exc_type}')
            print(f'exc_val: {exc_val}')
        time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start))
        self.__hp.update({'exc_val': exc_val, "duration": time_span})
        if self.__lp is not None:
            self.__write_log(**self.__hp)

    def __write_log(self, **kwargs):
        print('logging...')
        kwargs.update(self.__extra_lm)
        tools.write_log(self.__lp, **kwargs)

    def add_logMsg(self, **kwargs):
        self.__extra_lm = kwargs
