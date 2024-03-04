import time

import torch
from torchsummary import summary

import utils.func.log_tools as ltools
from utils.func import pytools


class Trainer:

    sn_range = ['no', 'entire', 'state']

    def __init__(self, datasource, hyper_parameters: dict, exp_no: int,
                 log_path: str = None, net_path: str = None,
                 print_net: bool = True, save_net: str = 'no'):
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
        self.__exp_no = exp_no
        self.__sn = save_net
        self.__pn = print_net
        self.datasource = datasource
        self.__net = None

    def __enter__(self):
        self.start = time.time()
        # print(
        #     f'\r---------------------------实验{self.__exp_no}号'
        #     f'---------------------------'
        # )
        for k, v in self.__hp.items():
            print(k + ': ' + str(v))
        print(
            '\r----------------------------------------------------------------'
        )
        return self.__hp.values()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """训练器对象的退出动作。
        进行日志编写以及网络保存操作。

        :param exc_type: 出现的异常类型
        :param exc_val: 出现的异常值
        :param exc_tb: 异常的路径回溯
        :return: None
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
                return
        # 记录时间信息
        time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start))
        self.__hp.update({'exp_no': self.__exp_no, "duration": time_span, "dataset": self.datasource.__name__})
        if self.__lp is not None:
            # 指定了日志路径，则进行日志记录
            self.__write_log(**self.__hp)
        # 保存训练生成的网络
        self.__save_net()

    def __write_log(self, **kwargs):
        kwargs.update(self.__extra_lm)
        ltools.write_log(self.__lp, **kwargs)
        print('已编写日志')

    def add_logMsg(self, mute=True, **kwargs):
        self.__extra_lm.update(kwargs)
        if not mute:
            print(self.__extra_lm)

    def __save_net(self) -> None:
        """
        保存当前网络net
        :return: None
        """
        if self.__np is None:
            print("未指定模型保存路径，不予保存模型！")
            return
        if not isinstance(self.__net, torch.nn.Module):
            print('训练器对象未得到训练网络对象，因此不予保存网络！')
            return
        if not pytools.check_para('save_net', self.__sn, self.sn_range):
            print('请检查setting.json中参数save_net设置是否正确，本次不予保存模型！')
            return
        if self.__sn == 'entire':
            torch.save(self.__net, self.__np + f'{self.__exp_no}.ptm')
        elif self.__sn == 'state':
            torch.save(self.__net.state_dict(), self.__np + f'{self.__exp_no}.ptsd')
        print('已保存网络')

    def __list_net(self, net, input_size, batch_size) -> None:
        """
        打印网络信息。
        :param net: 待打印的网络信息。
        :param input_size: 网络输入参数。
        :param batch_size: 训练的批量大小。
        :return: None
        """
        if self.__pn:
            try:
                summary(net, input_size=input_size, batch_size=batch_size)
            except RuntimeError as _:
                print(net)

    def register_net(self, net: torch.nn.Module):
        self.__net = net
        self.__list_net(
            net, (self.datasource.fea_channel, *net.required_shape),
            self.__hp['batch_size']
        )
