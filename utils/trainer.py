import time
import warnings

import torch
from torchsummary import summary

from utils import tools


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
        print(
            f'---------------------------EXP NO.{self.__exp_no}'
            f'---------------------------'
        )
        for k, v in self.__hp.items():
            print(k + ': ' + str(v))
        print(
            '----------------------------------------------------------------'
        )
        # TODO: 打印超参数
        return self.__hp.values()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 进行日志编写
        if exc_type is not None:
            print(f'exc_type: {exc_type}')
            print(f'exc_val: {exc_val}')
            self.__hp.update({'exc_val': exc_val})
        time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start))
        self.__hp.update({'exp_no': self.__exp_no, "duration": time_span, "dataset": self.datasource.__name__})
        if self.__lp is not None and exc_type != KeyboardInterrupt:
            self.__write_log(**self.__hp)
        # 保存训练生成的网络
        self.__save_net()

    def __write_log(self, **kwargs):
        kwargs.update(self.__extra_lm)
        tools.write_log(self.__lp, **kwargs)
        print('已编写日志')

    def add_logMsg(self, mute=True, **kwargs):
        self.__extra_lm = kwargs
        if not mute:
            print(self.__extra_lm)

    # def save_net(self, net: torch.nn.Module, exp_no: int, entire=False):
    # def save_net(self, net: torch.nn.Module, entire=False) -> None:
    #     """
    #     保存当前网络net
    #     :param net: 需要保存的网络
    #     :param entire: 是否保存整个网络
    #     :return: None
    #     """
    #     warnings.warn('将在未来的版本中删除，由settings.json中的save_net参数自动指示对网络的保存，由__save_net()进行该动作',
    #                   DeprecationWarning)
    #     if self.__np is None:
    #         print("未指定模型保存路径，不予保存模型！")
    #         return
    #     if entire:
    #         torch.save(net, self.__np + f'{self.__exp_no}.ptm')
    #     else:
    #         torch.save(net.state_dict(), self.__np + f'{self.__exp_no}.ptsd')

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
        if not tools.check_para('save_net', self.__sn, self.sn_range):
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
        self.__list_net(net, (self.datasource.img_channel, *net.required_shape), self.__hp['batch_size'])
