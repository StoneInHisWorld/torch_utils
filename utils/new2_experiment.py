import gc
import time
import warnings
from datetime import datetime

import torch

from .func import pytools as ptools


def _print_result(history, test_log):
    """输出训练日志
    将训练日志输出，如果有验证日志和测试日志则一并输出到命令行

    :param history: 训练（和验证）日志的历史记录对象
    :param test_log: 测试日志信息
    :return: 训练（和验证）日志聚合的日志信息
    """
    metrics_ls_msg = {}
    if len(history) <= 0:
        raise ValueError('历史记录对象为空！')
    # 输出训练部分的数据
    train_history = filter(lambda h: h[0].startswith('train_'), history)
    valid_history = list(filter(lambda h: h[0].startswith('valid_'), history))
    for name, log in train_history:
        # 输出训练信息，并记录
        print(f"{name.replace('train_', '训练')} = {log[-1]:.5f},", end=' ')
        metrics_ls_msg[name] = log[-1]
    # 输出验证部分的数据
    if len(valid_history) > 0:
        print('\b\b')
    for name, log in valid_history:
        # 输出验证信息，并记录
        print(f"{name.replace('valid_', '验证')} = {log[-1]:.5f},", end=' ')
        metrics_ls_msg[name] = log[-1]
    # 输出测试部分的数据
    print('\b\b')
    if test_log is not None:
        for k, v in test_log.items():
            print(f"{k.replace('test_', '测试')} = {v:.5f},", end=' ')
        print('\b\b')
    metrics_ls_msg.update(test_log)
    return metrics_ls_msg


class New2Experiment:
    """实验对象负责神经网络训练的相关周边操作，计时、显存监控、日志编写、网络持久化、历史趋势图绘制及保存"""

    def __init__(
            self, exp_no: int, datasource: type, net_type: type,
            hyper_parameters: dict, runtime_cfg: dict, is_train: bool
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
        :param metric_log_path: 日志所在路径
        :param net_path: 网络保存路径
        """
        # self.__extra_lm = {}
        self.__hp = hyper_parameters
        # self.__pp = plot_path
        # self.__mlp = metric_log_path
        # self.__plp = perf_log_path
        # self.__np = net_path
        self.__exp_no = exp_no
        self.cfg_dict = runtime_cfg
        self.dao_ds = datasource
        self.net_type = net_type
        self.is_train = is_train

    def __enter__(self):
        """训练对象的上下文管理进入方法
        负责进行计时、超参数打印以及显存监控。
        """
        # 计时开始
        self.start = time.time()
        # 提取框架固定配置参数
        self.__rearrange_config()
        # 打印本次训练超参数
        for k, v in self.__hp.items():
            print(k + ': ' + str(v))
        print(f'data_portion: {self.ds_config["data_portion"]}')
        print(
            '\r-----------------输入Ctrl+C即可终止本组超参数实验'
            '--------------------'
        )
        # 提取训练配置参数
        device = torch.device(self.cfg_dict['device'])
        cuda_memrecord = self.cfg_dict['cuda_memrecord']
        # 开启显存监控
        if device.type == 'cuda' and cuda_memrecord:
            torch.cuda.memory._record_memory_history(cuda_memrecord)
        elif device.type == 'cpu' and cuda_memrecord:
            warnings.warn(f'运行设备为{device}，不支持显存监控！请使用支持CUDA的处理机，或者设置cuda_memrecord为false')
        # 加载数据集
        self.data = self.__build_dao_ds(self.__hp)
        net_builder = self.__build_net_builder()
        self.trainer = self.__build_trainer(net_builder, self.data.get_criterion_a(), self.t_kwargs)
        return self.data, net_builder, self.__hp

    def __exit__(self, exc_type, exc_val, exc_tb):
        """训练器对象的退出动作。
        进行日志编写以及网络保存操作。

        :param exc_type: 出现的异常类型
        :param exc_val: 出现的异常值
        :param exc_tb: 异常的路径回溯
        """
        # 进行日志编写
        # basic_perf_log = self.basic_perf_log if hasattr(self, 'basic_perf_log') else {}
        if exc_type is not None:
            if exc_type == MemoryError:
                gc.collect()
            elif exc_type == KeyboardInterrupt:
                # 键盘中断则什么也不做
                print('该组超参数实验被中断！')
                try:
                    for i in range(10, 0, -1):
                        print(f'\r将在{i}秒后继续进行下一组超参数实验，再中断一次即可终止整个程序', end='', flush=True)
                        time.sleep(1)
                    return True
                except KeyboardInterrupt:
                    raise KeyboardInterrupt('实验被中断！')
            # # 出现异常则记录
            # print(f'\n出现{exc_type}异常\n描述为{exc_val}')
        # 编辑日志条目，加入数据量、数据形状和显存消耗等内容
        basic_metric_log, basic_perf_log = {}, {}
        if self.device != torch.device('cpu') and self.cfg_dict["cuda_memrecord"]:
            basic_perf_log.update(
                max_GPUmemory_allocated=torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
                max_GPUmemory_reserved=torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
            )
            # 刷新显存消耗
            torch.cuda.reset_peak_memory_stats(self.device)
        if exc_type is None:
            basic_metric_log.update(
                **self.__hp, exp_no=self.__exp_no, dataset=self.dao_ds.__name__,
                data_portion=self.ds_config['data_portion'],
                f_req_shp=self.ds_config['f_req_shp'],
                l_req_shp=self.ds_config['l_req_shp'],
            )
            metric_log, perf_log = self.__register_result()
            basic_metric_log.update(metric_log)
            basic_perf_log.update(perf_log)
        basic_perf_log.update(
            exp_no=self.__exp_no, n_workers=self.cfg_dict['n_workers'],
            time_stamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            duration=time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start)),
            n_workers_for_trainer=self.t_kwargs['n_workers'],
            n_workers_for_dloader=self.dl_config['num_workers'],
            n_workers_for_ds=self.ds_config['n_workers'],
            exc_type=exc_type, exc_val=exc_val
        )
        self.metric_log, self.perf_log = basic_metric_log, basic_perf_log
        return None

    def __rearrange_config(self):
        # 数据集参数
        self.ds_config = self.cfg_dict.pop("ds_kwargs")
        # 数据迭代器参数
        self.dl_config = self.cfg_dict.pop("dl_kwargs")
        k = self.__hp.pop("k")
        batch_size = self.__hp.pop("batch_size")
        self.dl_config["k"] = k
        self.dl_config["batch_size"] = batch_size
        # 网络创建器参数
        self.nb_kwargs = self.cfg_dict.pop("nb_kwargs")
        # 训练器参数
        self.t_kwargs = self.cfg_dict.pop("t_kwargs")
        self.t_kwargs["batch_size"] = batch_size
        if self.is_train:
            self.t_kwargs["k"] = k
            self.t_kwargs["n_epochs"] = self.__hp.pop("n_epochs")

    def __build_dao_ds(self, hyper_params):
        pos_only_args, pos_or_kwargs, kwargs_only, _ = ptools.get_signature(self.dao_ds)
        args = [hyper_params.pop(poa) for poa in pos_only_args + pos_or_kwargs]
        kwargs = {ko: hyper_params.pop(ko) for ko in kwargs_only}
        return self.dao_ds(*args, **kwargs, module=self.net_type, is_train=self.is_train,
                           ds_config=self.ds_config, dl_config=self.dl_config)

    def __build_net_builder(self):
        from networks import NetBuilder

        return NetBuilder(self.net_type, self.nb_kwargs)

    def __build_trainer(self, net_builder, criteria_fns, t_kwargs):
        from networks import New2Trainer

        trainer = New2Trainer(net_builder, criteria_fns, t_kwargs)
        return trainer

    def __register_result(self):
        """根据训练历史记录进行输出，并进行日志参数的记录。
        在神经网络训练完成后，需要调用本函数将结果注册到超参数控制台。
        :param net: 训练完成的网络对象
        :param train_logs: 训练历史记录
        :param test_logs: 测试记录
        :return: None
        """
        # 保存训练生成的网络
        self.net = self.trainer.module if hasattr(self.trainer, "module") else None
        # 提取训练的历史记录
        self.train_mlog, train_dlog = self.train_histories if hasattr(self, "train_histories") else ({}, {})
        test_mlog, test_dlog = self.test_histories if hasattr(self, "test_histories") else ({}, {})
        # 计算花费时间项
        perf_log = {**{
            k + "(s)": sum(log) / len(log) for k, log in train_dlog
        }, **{
            k + "(s)": log for k, log in test_dlog.items()
        }}
        # 将最后一个世代的信息计算并打印在命令行中
        metric_log = _print_result(self.train_mlog, test_mlog)
        return metric_log, perf_log
        # # 绘制历史趋势图
        # with warnings.catch_warnings():
        #     # 忽视空图图例警告
        #     warnings.simplefilter('ignore', category=UserWarning)
        #     self.__plot_history(self.train_mlog, **plot_kwargs)

    # def __plot_history(self, history, **plot_kwargs) -> None:
    #     """绘制历史趋势图
    #     根据需要绘制历史趋势图，有三种模式可选，模式选择由动态运行参数plot_history指定：
    #     plot： 绘制图像，但不进行保存
    #     save：绘制图像且保存，需要指定保存路径self.__pp，保存图片名为(实验编号.jpg)
    #     no：不绘制历史趋势图
    #
    #     :param history: 历史记录对象
    #     :param plot_kwargs: 历史趋势图绘制关键字参数
    #     :return: None
    #     """
    #     # 检查参数设置
    #     cfg_range = ['plot', 'save', 'no']
    #     cfg = self.cfg_dict['plot_history']
    #     if not ptools.check_para('plot_history', cfg, cfg_range):
    #         warnings.warn(
    #             '请检查setting.json中参数plot_history设置是否正确，本次不予绘制历史趋势图！',
    #             UserWarning
    #         )
    #         return
    #     if cfg == 'no':
    #         return
    #     if self.__pp is None:
    #         warnings.warn('未指定绘图路径，不予保存历史趋势图！')
    #     savefig_as = None if self.__pp is None or cfg == 'plot' else (
    #             self.__pp + str(self.__exp_no) + '.jpg')
    #     # 绘图
    #     ltools.plot_history(
    #         history, mute=self.cfg_dict['plot_mute'],
    #         title='EXP NO.' + str(self.__exp_no),
    #         savefig_as=savefig_as, **plot_kwargs
    #     )

    def train(self, transit_fn=None, **dl_kwargs):
        # 将数据集转化为迭代器
        train_iter = self.data.to_dataloaders(True, transit_fn, **dl_kwargs)
        self.train_histories = self.trainer.train(train_iter)

    def test(self, transit_fn=None, **dl_kwargs):
        # 将数据集转化为迭代器
        test_iter = self.data.to_dataloaders(False, transit_fn, **dl_kwargs)
        self.test_histories = self.trainer.test(test_iter)


    @property
    def device(self):
        return torch.device(self.cfg_dict['device'])
